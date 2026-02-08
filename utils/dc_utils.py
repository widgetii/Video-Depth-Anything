# This file is originally from DepthCrafter/depthcrafter/utils.py at main · Tencent/DepthCrafter
# SPDX-License-Identifier: MIT License license
#
# This file may have been modified by ByteDance Ltd. and/or its affiliates on [date of modification]
# Original file is released under [ MIT License license], with the full license text available at [https://github.com/Tencent/DepthCrafter?tab=License-1-ov-file].
import numpy as np
import matplotlib.cm as cm
import imageio
import os
import subprocess
import shutil
import psutil

try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
    try:
        from decord import gpu as decord_gpu
        DECORD_GPU_AVAILABLE = True
    except ImportError:
        DECORD_GPU_AVAILABLE = False
except:
    import cv2
    DECORD_AVAILABLE = False
    DECORD_GPU_AVAILABLE = False

def _log_mem(stage: str):
    proc = psutil.Process(os.getpid())
    rss_gb = proc.memory_info().rss / (1024 ** 3)
    print(f"[MEM][dc_utils] {stage}: RSS={rss_gb:.2f} GB", flush=True)

def ensure_even(value):
    return value if value % 2 == 0 else value + 1


def _make_decord_ctx(device='cpu'):
    """Create decord context: GPU (NVDEC) if available and requested, else CPU."""
    if device != 'cpu' and DECORD_GPU_AVAILABLE:
        try:
            import torch
            gpu_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
            ctx = decord_gpu(gpu_id)
            print(f"[decord] Using GPU decode (NVDEC) on device {gpu_id}", flush=True)
            return ctx
        except Exception as e:
            print(f"[decord] GPU decode failed ({e}), falling back to CPU", flush=True)
    return cpu(0)


class LazyVideoFrames:
    """Lazy video frame reader that decodes frames on demand.
    
    Supports indexed access (reader[i]) and iteration.
    Frames are decoded one at a time and NOT cached, so memory stays flat.
    Uses NVDEC (GPU) decoding when available.
    """
    def __init__(self, video_path, frames_idx, width, height, fps, device='cpu'):
        self._video_path = video_path
        self._frames_idx = frames_idx  # list of original frame indices to use
        self._width = width
        self._height = height
        self.fps = fps
        # Keep a persistent VideoReader for sequential access
        if DECORD_AVAILABLE:
            ctx = _make_decord_ctx(device)
            try:
                self._vid = VideoReader(video_path, ctx=ctx, width=width, height=height)
            except Exception as e:
                if device != 'cpu':
                    print(f"[decord] GPU VideoReader failed ({e}), falling back to CPU", flush=True)
                    self._vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)
                else:
                    raise
        else:
            self._vid = None
    
    def __len__(self):
        return len(self._frames_idx)
    
    @property
    def shape(self):
        """Return (N, H, W, C) like a numpy array for compatibility."""
        return (len(self._frames_idx), self._height, self._width, 3)
    
    def __getitem__(self, idx):
        """Decode and return a single frame as numpy uint8 array (H, W, 3)."""
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self._frames_idx)))
            return [self._get_single(i) for i in indices]
        if idx < 0:
            idx += len(self._frames_idx)
        return self._get_single(idx)
    
    def _get_single(self, idx):
        real_idx = self._frames_idx[idx]
        if DECORD_AVAILABLE:
            return self._vid[real_idx].asnumpy()
        else:
            cap = cv2.VideoCapture(self._video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, real_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise RuntimeError(f"Failed to read frame {real_idx} from {self._video_path}")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self._width != frame.shape[1] or self._height != frame.shape[0]:
                frame = cv2.resize(frame, (self._width, self._height))
            return frame
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def get_video_info(video_path, process_length, target_fps=-1, max_res=-1, device='cpu'):
    """Get video metadata and create a lazy frame reader without decoding all frames.
    
    Uses NVDEC (GPU) decoding when device != 'cpu' and decord GPU is available.
    
    Returns: (lazy_reader, fps)
    """
    if DECORD_AVAILABLE:
        vid = VideoReader(video_path, ctx=cpu(0))
        original_height, original_width = vid[0].shape[:2]
        height = original_height
        width = original_width
        if max_res > 0 and max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = ensure_even(round(original_height * scale))
            width = ensure_even(round(original_width * scale))

        fps = vid.get_avg_fps() if target_fps == -1 else target_fps
        stride = round(vid.get_avg_fps() / fps)
        stride = max(stride, 1)
        frames_idx = list(range(0, len(vid), stride))
        if process_length != -1 and process_length < len(frames_idx):
            frames_idx = frames_idx[:process_length]
        
        del vid
    else:
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        height, width = original_height, original_width
        if max_res > 0 and max(original_height, original_width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = round(original_height * scale)
            width = round(original_width * scale)

        fps = original_fps if target_fps < 0 else target_fps
        stride = max(round(original_fps / fps), 1)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_idx = list(range(0, total_frames, stride))
        if process_length != -1 and process_length < len(frames_idx):
            frames_idx = frames_idx[:process_length]
        cap.release()

    _log_mem(f'video info read — {len(frames_idx)} frames selected, {width}x{height}')
    return LazyVideoFrames(video_path, frames_idx, width, height, fps, device=device), fps


def save_source_video_ffmpeg(input_path, output_path, fps=-1, max_res=-1):
    """Re-encode source video using ffmpeg directly — no Python decode needed.
    
    Applies fps and resolution changes via ffmpeg filters, using NVENC if available.
    """
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        print("[WARN] ffmpeg not found, skipping source video save", flush=True)
        return False
    
    vf_filters = []
    if max_res > 0:
        # Scale so longest side <= max_res, keep aspect, ensure even dimensions
        vf_filters.append(
            f"scale='if(gt(iw,ih),min({max_res},iw),-2)':'if(gt(ih,iw),min({max_res},ih),-2)'"
        )
        # Ensure both dimensions are even
        vf_filters.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")
    
    cmd = [ffmpeg, '-y', '-i', input_path]
    if fps > 0:
        cmd += ['-r', str(fps)]
    if vf_filters:
        cmd += ['-vf', ','.join(vf_filters)]
    cmd += ['-c:v', 'libx264', '-crf', '18', '-an', '-movflags', '+faststart', output_path]
    
    print(f"[ffmpeg] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ffmpeg] FAILED: {result.stderr[-500:]}", flush=True)
        return False
    _log_mem('ffmpeg source video saved')
    return True

def read_video_frames(video_path, process_length, target_fps=-1, max_res=-1, max_batch_size=64):
    if DECORD_AVAILABLE:
        vid = VideoReader(video_path, ctx=cpu(0))
        original_height, original_width = vid.get_batch([0]).shape[1:3]
        height = original_height
        width = original_width
        if max_res > 0 and max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = ensure_even(round(original_height * scale))
            width = ensure_even(round(original_width * scale))

        vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

        fps = vid.get_avg_fps() if target_fps == -1 else target_fps
        stride = round(vid.get_avg_fps() / fps)
        stride = max(stride, 1)
        frames_idx = list(range(0, len(vid), stride))
        if process_length != -1 and process_length < len(frames_idx):
            frames_idx = frames_idx[:process_length]
        # Load in batches to avoid OOM instead of one giant get_batch call
        frames_list = []
        for i in range(0, len(frames_idx), max_batch_size):
            batch_idx = frames_idx[i : i + max_batch_size]
            batch = vid.get_batch(batch_idx).asnumpy()
            frames_list.append(batch)
        frames = np.concatenate(frames_list, axis=0)
    else:
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        if max_res > 0 and max(original_height, original_width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = round(original_height * scale)
            width = round(original_width * scale)

        fps = original_fps if target_fps < 0 else target_fps

        stride = max(round(original_fps / fps), 1)

        frames = []
        frame_count = 0
        selected_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % stride == 0:
                if process_length > 0 and selected_count >= process_length:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                if max_res > 0 and max(original_height, original_width) > max_res:
                    frame = cv2.resize(frame, (width, height))  # Resize frame
                frames.append(frame)
                selected_count += 1
            frame_count += 1
        cap.release()
        frames = np.stack(frames, axis=0)

    return frames, fps


def save_video(frames, output_video_path, fps=10, is_depths=False, grayscale=False):
    writer = imageio.get_writer(output_video_path, fps=fps, macro_block_size=1, codec='libx264', ffmpeg_params=['-crf', '18'])
    if is_depths:
        colormap = np.array(cm.get_cmap("inferno").colors)
        # For depths we need global min/max — requires numpy array
        if not isinstance(frames, np.ndarray):
            frames = np.stack(list(frames), axis=0)
        d_min, d_max = frames.min(), frames.max()
        for i in range(frames.shape[0]):
            depth = frames[i]
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            depth_vis = (colormap[depth_norm] * 255).astype(np.uint8) if not grayscale else depth_norm
            writer.append_data(depth_vis)
    else:
        # Stream frames one at a time — works with LazyVideoFrames, list, or ndarray
        for i, frame in enumerate(frames):
            writer.append_data(np.asarray(frame))
            if i % 500 == 0:
                _log_mem(f'save_video frame {i}/{len(frames)}')

    writer.close()
