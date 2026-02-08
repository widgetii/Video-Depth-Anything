# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose
import cv2
from tqdm import tqdm
import numpy as np
import gc
import os
import psutil

def _log_mem(stage: str):
    proc = psutil.Process(os.getpid())
    rss_gb = proc.memory_info().rss / (1024 ** 3)
    print(f"[MEM][infer] {stage}: RSS={rss_gb:.2f} GB", flush=True)

from .dinov2 import DINOv2
from .dpt_temporal import DPTHeadTemporal
from .util.transform import Resize, NormalizeImage, PrepareForNet

from utils.util import compute_scale_and_shift, get_interpolate_frames

# infer settings, do not change
INFER_LEN = 32
OVERLAP = 10
KEYFRAMES = [0,12,24,25,26,27,28,29,30,31]
INTERP_LEN = 8

class VideoDepthAnything(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False,
        num_frames=32,
        pe='ape',
        metric=False,
    ):
        super(VideoDepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23]
        }

        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)

        self.head = DPTHeadTemporal(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken, num_frames=num_frames, pe=pe)
        self.metric = metric

    def forward(self, x):
        B, T, C, H, W = x.shape
        patch_h, patch_w = H // 14, W // 14
        features = self.pretrained.get_intermediate_layers(x.flatten(0,1), self.intermediate_layer_idx[self.encoder], return_class_token=True)
        depth = self.head(features, patch_h, patch_w, T)[0]
        depth = F.interpolate(depth, size=(H, W), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        return depth.squeeze(1).unflatten(0, (B, T)) # return shape [B, T, H, W]

    def infer_video_depth(self, frames, target_fps, input_size=518, device='cuda', fp32=False):
        """Streaming inference: yields batches of finalized aligned depth frames (float16).
        
        Each yield is a list of numpy arrays (H, W) in float16.
        Frames are yielded as soon as they are finalized (will not change).
        Memory usage stays nearly constant regardless of video length.
        """
        frame_height, frame_width = frames[0].shape[:2]
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14

        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        org_video_len = len(frames)
        frame_step = INFER_LEN - OVERLAP
        align_len = OVERLAP - INTERP_LEN
        kf_align_list = KEYFRAMES[:align_len]

        # Read last frame once for padding
        last_frame = np.array(frames[org_video_len - 1])

        # Pre-compute transformed shape from first frame
        first_frame = np.array(frames[0])
        first_transformed = torch.from_numpy(transform({'image': first_frame.astype(np.float32) / 255.0})['image'])
        transformed_shape = first_transformed.shape
        del first_frame, first_transformed

        _log_mem(f'starting streaming inference — {org_video_len} frames, window={INFER_LEN}, step={frame_step}')

        # Aligned buffer: only holds frames that might still be modified (up to INTERP_LEN)
        aligned_tail = []  # last INTERP_LEN frames that may be modified by next chunk
        ref_align = []
        pre_input = None
        total_yielded = 0
        chunk_idx = 0

        for frame_id in tqdm(range(0, org_video_len, frame_step)):
            # --- INFERENCE for this chunk ---
            cur_list = []
            decode_start = OVERLAP if pre_input is not None else 0
            for i in range(INFER_LEN):
                if i < decode_start:
                    cur_list.append(torch.zeros(1, 1, *transformed_shape))
                else:
                    abs_idx = frame_id + i
                    if abs_idx < org_video_len:
                        raw_frame = np.array(frames[abs_idx])
                    else:
                        raw_frame = last_frame.copy()
                    cur_list.append(torch.from_numpy(
                        transform({'image': raw_frame.astype(np.float32) / 255.0})['image']
                    ).unsqueeze(0).unsqueeze(0))

            cur_input = torch.cat(cur_list, dim=1).to(device)
            if pre_input is not None:
                cur_input[:, :OVERLAP, ...] = pre_input[:, KEYFRAMES, ...]

            with torch.no_grad():
                with torch.autocast(device_type=device, enabled=(not fp32)):
                    depth = self.forward(cur_input)

            depth = depth.to(cur_input.dtype)
            depth = F.interpolate(depth.flatten(0, 1).unsqueeze(1),
                                  size=(frame_height, frame_width), mode='bilinear', align_corners=True)
            chunk_depths = [depth[i][0].cpu().numpy().astype(np.float16) for i in range(depth.shape[0])]
            pre_input = cur_input

            # --- ALIGNMENT for this chunk ---
            if chunk_idx == 0:
                # First chunk: no alignment needed
                ref_align = [chunk_depths[kf_id].astype(np.float32) for kf_id in kf_align_list]
                # Yield everything except the last INTERP_LEN (those may be modified)
                finalized = chunk_depths[:INFER_LEN - INTERP_LEN]
                aligned_tail = chunk_depths[INFER_LEN - INTERP_LEN:INFER_LEN]
            else:
                # Compute scale and shift
                curr_align = [chunk_depths[i].astype(np.float32) for i in range(len(kf_align_list))]
                if self.metric:
                    scale, shift = 1.0, 0.0
                else:
                    scale, shift = compute_scale_and_shift(
                        np.concatenate(curr_align),
                        np.concatenate(ref_align),
                        np.concatenate(np.ones_like(ref_align) == 1))

                # Interpolate overlap zone
                pre_depth_list = [d.astype(np.float32) for d in aligned_tail]
                post_depth_list = [chunk_depths[align_len + i].astype(np.float32) for i in range(INTERP_LEN)]
                for i in range(len(post_depth_list)):
                    post_depth_list[i] = post_depth_list[i] * scale + shift
                    post_depth_list[i][post_depth_list[i] < 0] = 0
                interp_result = get_interpolate_frames(pre_depth_list, post_depth_list)
                interpolated = [d.astype(np.float16) for d in interp_result]

                # Apply scale+shift to remaining frames
                new_frames = []
                for i in range(OVERLAP, INFER_LEN):
                    d = chunk_depths[i].astype(np.float32) * scale + shift
                    d[d < 0] = 0
                    new_frames.append(d.astype(np.float16))

                # All frames in this chunk's output: interpolated + new_frames
                all_new = interpolated + new_frames
                # Finalized = everything except last INTERP_LEN
                finalized = all_new[:len(all_new) - INTERP_LEN]
                aligned_tail = all_new[len(all_new) - INTERP_LEN:]

                # Update ref_align
                ref_align = ref_align[:1]
                for kf_id in kf_align_list[1:]:
                    d = chunk_depths[kf_id].astype(np.float32) * scale + shift
                    d[d < 0] = 0
                    ref_align.append(d)

            # --- YIELD finalized frames ---
            # Don't yield more than org_video_len total
            for f in finalized:
                if total_yielded < org_video_len:
                    yield f
                    total_yielded += 1

            chunk_idx += 1

            if frame_id % (frame_step * 10) == 0:
                _log_mem(f'streaming frame_id={frame_id}/{org_video_len}, yielded={total_yielded}')

        # Flush remaining tail frames
        for f in aligned_tail:
            if total_yielded < org_video_len:
                yield f
                total_yielded += 1

        del last_frame
        gc.collect()
        _log_mem(f'streaming done — yielded {total_yielded} frames total')

