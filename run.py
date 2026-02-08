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
import argparse
import numpy as np
import os
import sys
import torch
import psutil

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import get_video_info, save_video

def log_memory(stage: str):
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    rss_gb = mem.rss / (1024 ** 3)
    vms_gb = mem.vms / (1024 ** 3)
    print(f"[MEM] {stage}: RSS={rss_gb:.2f} GB, VMS={vms_gb:.2f} GB", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_video', type=str, default='./assets/example_videos/davis_rollercoaster.mp4')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--metric', action='store_true', help='use metric model')
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--save_npz', action='store_true', help='save depths as npz')
    parser.add_argument('--save_exr', action='store_true', help='save depths as exr')
    parser.add_argument('--global_normalize', action='store_true',
                        help='use global min/max normalization for depth vis (two-pass, smoother but slower)')
    parser.add_argument('--focal-length-x', default=470.4, type=float,
                        help='Focal length along the x-axis.')
    parser.add_argument('--focal-length-y', default=470.4, type=float,
                        help='Focal length along the y-axis.')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder], metric=args.metric)
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/{checkpoint_name}_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()
    log_memory('model loaded')

    import gc
    import tempfile

    video_name = os.path.basename(args.input_video)
    os.makedirs(args.output_dir, exist_ok=True)
    depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_vis.mp4')

    # Lazy reader with GPU decode (NVDEC) — decodes and scales on-the-fly, nothing stored in RAM
    lazy_frames, target_fps = get_video_info(args.input_video, args.max_len, args.target_fps, args.max_res, device=DEVICE)
    log_memory(f'video info read — {len(lazy_frames)} frames at {target_fps} fps (no frames decoded yet)')

    import imageio
    import matplotlib.cm as cm

    exr_dir = None
    if args.save_exr:
        exr_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_exr')
        os.makedirs(exr_dir, exist_ok=True)
        import OpenEXR
        import Imath

    npz_dir = None
    if args.save_npz:
        npz_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_npy')
        os.makedirs(npz_dir, exist_ok=True)

    colormap = np.array(cm.get_cmap("inferno").colors)
    frame_idx = 0

    def _write_raw_outputs(depth_frame, idx):
        """Write EXR and npy outputs for a single frame (normalization-independent)."""
        if exr_dir is not None:
            output_exr = f"{exr_dir}/frame_{idx:05d}.exr"
            header = OpenEXR.Header(depth_frame.shape[1], depth_frame.shape[0])
            header["channels"] = {
                "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
            }
            exr_file = OpenEXR.OutputFile(output_exr, header)
            exr_file.writePixels({"Z": depth_frame.astype(np.float32).tobytes()})
            exr_file.close()
        if npz_dir is not None:
            np.save(f"{npz_dir}/frame_{idx:05d}.npy", depth_frame)

    def _depth_to_vis(depth_frame, d_min, d_max):
        """Normalize a depth frame and apply colormap."""
        d_range = d_max - d_min
        if d_range == 0:
            d_range = 1.0
        depth_norm = ((depth_frame.astype(np.float32) - d_min) / d_range * 255).astype(np.uint8)
        if not args.grayscale:
            return (colormap[depth_norm] * 255).astype(np.uint8)
        return depth_norm

    if args.global_normalize:
        # ── Two-pass: stream raw depths to temp file, then visualize with global min/max ──
        # Temp file stores raw float16 depths sequentially (~1.7 MB/frame for 720x1280).
        depth_tmp = tempfile.NamedTemporaryFile(suffix='.raw', delete=False)
        depth_shape = None
        depth_global_min = float('inf')
        depth_global_max = float('-inf')

        for depth_frame in video_depth_anything.infer_video_depth(lazy_frames, target_fps,
                                                                   input_size=args.input_size,
                                                                   device=DEVICE, fp32=args.fp32):
            if depth_shape is None:
                depth_shape = depth_frame.shape

            d_min = float(depth_frame.min())
            d_max = float(depth_frame.max())
            depth_global_min = min(depth_global_min, d_min)
            depth_global_max = max(depth_global_max, d_max)

            depth_tmp.write(depth_frame.tobytes())
            _write_raw_outputs(depth_frame, frame_idx)

            frame_idx += 1
            if frame_idx % 500 == 0:
                log_memory(f'pass 1: streamed {frame_idx} frames')

        depth_tmp.close()
        log_memory(f'pass 1 done — {frame_idx} frames, range=[{depth_global_min:.4f}, {depth_global_max:.4f}]')

        # Free model and reader before pass 2
        del video_depth_anything, lazy_frames
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        log_memory('model freed')

        # Pass 2: read back and write vis with global normalization
        vis_writer = imageio.get_writer(depth_vis_path, fps=target_fps, macro_block_size=1,
                                         codec='libx264', ffmpeg_params=['-crf', '18'])
        frame_nbytes = int(np.prod(depth_shape)) * 2  # float16 = 2 bytes

        with open(depth_tmp.name, 'rb') as f:
            for i in range(frame_idx):
                raw = f.read(frame_nbytes)
                depth_frame = np.frombuffer(raw, dtype=np.float16).reshape(depth_shape)
                vis_writer.append_data(_depth_to_vis(depth_frame, depth_global_min, depth_global_max))

        vis_writer.close()
        os.unlink(depth_tmp.name)
        log_memory('pass 2 done — vis video written with global normalization')

    else:
        # ── Single-pass (default): per-frame normalization, no temp file ──
        vis_writer = imageio.get_writer(depth_vis_path, fps=target_fps, macro_block_size=1,
                                         codec='libx264', ffmpeg_params=['-crf', '18'])

        for depth_frame in video_depth_anything.infer_video_depth(lazy_frames, target_fps,
                                                                   input_size=args.input_size,
                                                                   device=DEVICE, fp32=args.fp32):
            d_min = float(depth_frame.min())
            d_max = float(depth_frame.max())

            vis_writer.append_data(_depth_to_vis(depth_frame, d_min, d_max))
            _write_raw_outputs(depth_frame, frame_idx)

            frame_idx += 1
            if frame_idx % 500 == 0:
                log_memory(f'streamed {frame_idx} frames')

        vis_writer.close()
        log_memory(f'streaming done — {frame_idx} depth frames written')

        del video_depth_anything, lazy_frames
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        log_memory('model freed')

    if args.metric:
        import open3d as o3d

        metric_frames, _ = get_video_info(args.input_video, args.max_len, args.target_fps, args.max_res, device=DEVICE)
        # Load depth frames from saved npy files if available, otherwise warn
        if npz_dir is not None:
            width, height = None, None
            for i in range(frame_idx):
                depth = np.load(f"{npz_dir}/frame_{i:05d}.npy")
                if width is None:
                    width, height = depth.shape[-1], depth.shape[-2]
                    x, y = np.meshgrid(np.arange(width), np.arange(height))
                    x = (x - width / 2) / args.focal_length_x
                    y = (y - height / 2) / args.focal_length_y
                color_image = metric_frames[i]
                z = depth.astype(np.float32)
                points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
                colors = np.array(color_image).reshape(-1, 3) / 255.0
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(os.path.join(args.output_dir, 'point' + str(i).zfill(4) + '.ply'), pcd)
        else:
            print("[WARN] --metric requires depth data. Use --save_npz or --save_exr to enable point cloud export.", flush=True)
