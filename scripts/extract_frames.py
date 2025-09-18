#!/usr/bin/env python3
"""
extract_frames.py

Extracts frames from videos, resizes/crops to a target size, and chunks them into
overlapping clips. Produces a directory tree:

out_root/
  <video_name>/
    frames/                # (optional) all sampled frames
    clips/
      clip_0000/
        frame_000000.jpg
        ...
      clip_0001/
        ...
    clips.json             # manifest describing clip start/end times (sec), fps, etc.

Requirements:
  - opencv-python
  - numpy
  - tqdm

Example:
  python extract_frames.py \
    --in data/video_raw \
    --out data/video \
    --fps 16 --size 128 \
    --clip-seconds 3 --hop-seconds 1
"""
from __future__ import annotations
import argparse
import json
import math
import os
import shutil
import sys
from enum import Enum
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional, Union, cast
import cv2
import numpy as np
from tqdm import tqdm

# Constants
class ImageFormat(str, Enum):
    JPG = ".jpg"
    PNG = ".png"
    WEBP = ".webp"

DEFAULT_IMG_FORMAT = ImageFormat.JPG
IMG_QUALITY = 95  # Quality for JPEG/WEBP (1-100)

# Type aliases
ImageArray = np.ndarray[Any, np.dtype[np.uint8]]
ImageDims = Tuple[int, int]

def parse_hw(size: str) -> ImageDims:
    """Parse size string into height and width dimensions.
    
    Args:
        size: Either a single number (e.g., '128') or 'HxW' format (e.g., '128x128')
        
    Returns:
        Tuple of (height, width)
        
    Raises:
        ValueError: If the input format is invalid
    """
    try:
        if "x" in size.lower():
            h, w = size.lower().split("x")
            return int(h.strip()), int(w.strip())
        s = int(size)
        return s, s
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Invalid size format: {size}. Expected 'HxW' or single number") from e

def ensure_dir(p: Union[Path, str]) -> None:
    """Create directory if it doesn't exist.
    
    Args:
        p: Directory path as Path or string
        
    Raises:
        PermissionError: If directory creation fails due to permissions
        OSError: For other filesystem-related errors
    """
    path = Path(p) if isinstance(p, str) else p
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(f"Permission denied when creating directory: {path}") from e
    except OSError as e:
        raise OSError(f"Failed to create directory {path}: {str(e)}") from e

def link_or_copy(src: Union[Path, str], dst: Union[Path, str]) -> None:
    """Create a hardlink if possible, else copy.
    
    Args:
        src: Source file path
        dst: Destination file path
        
    Raises:
        FileNotFoundError: If source file doesn't exist
        OSError: If both hardlink and copy operations fail
    """
    src_path = Path(src) if isinstance(src, str) else src
    dst_path = Path(dst) if isinstance(dst, str) else dst
    
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    
    # Ensure destination directory exists
    ensure_dir(dst_path.parent)
    
    # Remove destination if it exists
    if dst_path.exists():
        try:
            dst_path.unlink()
        except OSError as e:
            raise OSError(f"Failed to remove existing destination {dst_path}: {e}") from e
    
    # Try hardlink first, fall back to copy
    try:
        os.link(src_path, dst_path)
    except (OSError, AttributeError) as link_error:
        # Fall back to copy if hardlink fails
        try:
            shutil.copy2(src_path, dst_path)
        except (shutil.SameFileError, PermissionError, OSError) as copy_error:
            raise OSError(
                f"Failed to copy {src_path} to {dst_path}: {str(copy_error)}"
            ) from copy_error

def center_resize_crop(
    img: ImageArray, 
    out_h: int, 
    out_w: int,
    interpolation: int = cv2.INTER_AREA
) -> ImageArray:
    """Resize (keep aspect) then center-crop to out_h x out_w.
    
    Args:
        img: Input image as numpy array (H, W, C) or (H, W)
        out_h: Target height in pixels
        out_w: Target width in pixels
        interpolation: OpenCV interpolation method (default: INTER_AREA)
        
    Returns:
        Resized and cropped image as uint8 numpy array
        
    Raises:
        ValueError: If input image is invalid or dimensions are invalid
    """
    if not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("Input image must be a non-empty numpy array")
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f"Output dimensions must be positive, got {out_h}x{out_w}")
    
    # Convert grayscale to 3-channel if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Remove alpha channel if present
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"Input image has invalid dimensions: {w}x{h}")
    
    # Calculate scale and new dimensions
    scale = min(out_h / h, out_w / w)  # Use min to ensure we don't crop too much
    nh, nw = int(round(h * scale)), int(round(w * scale))
    
    # Resize with anti-aliasing
    img_resized = cv2.resize(
        img, 
        (nw, nh), 
        interpolation=interpolation
    )
    
    # Center crop
    y0 = max(0, (nh - out_h) // 2)
    x0 = max(0, (nw - out_w) // 2)
    cropped = img_resized[y0:y0+out_h, x0:x0+out_w]
    
    # Ensure output has exactly the requested dimensions
    if cropped.shape[0] != out_h or cropped.shape[1] != out_w:
        cropped = cv2.resize(cropped, (out_w, out_h), interpolation=interpolation)
    
    return cropped.astype(np.uint8)

def sample_indices(n_src_frames: int, src_fps: float, tgt_fps: float) -> List[int]:
    """Calculate frame indices to sample for target FPS.
    
    Args:
        n_src_frames: Number of frames in source video
        src_fps: Source frames per second
        tgt_fps: Target frames per second (0 = use all frames)
        
    Returns:
        List of frame indices to sample
        
    Raises:
        ValueError: If FPS values are invalid
    """
    if n_src_frames <= 0:
        return []
    if src_fps <= 0:
        raise ValueError(f"Source FPS must be positive, got {src_fps}")
    if tgt_fps <= 0:
        return list(range(n_src_frames))
        
    try:
        step = src_fps / tgt_fps
        if step <= 0:
            return list(range(n_src_frames))
            
        idxs = [
            int(round(i * step)) 
            for i in range(int(math.floor((n_src_frames - 1) / step)) + 1)
        ]
        return [i for i in idxs if i < n_src_frames]
    except (ValueError, ZeroDivisionError) as e:
        raise ValueError(
            f"Error calculating sample indices with src_fps={src_fps}, "
            f"tgt_fps={tgt_fps}: {str(e)}"
        ) from e

def chunk_ranges(
    n_frames: int, 
    clip_len: int, 
    hop: int
) -> List[Tuple[int, int]]:
    """Split frame indices into overlapping or non-overlapping clips.
    
    Args:
        n_frames: Total number of frames
        clip_len: Length of each clip in frames
        hop: Number of frames to move between clips
        
    Returns:
        List of (start_idx, end_idx) tuples for each clip
        
    Raises:
        ValueError: If clip_len or hop are invalid
    """
    if n_frames <= 0:
        return []
    if clip_len <= 0:
        raise ValueError(f"Clip length must be positive, got {clip_len}")
    if hop <= 0:
        raise ValueError(f"Hop size must be positive, got {hop}")
        
    ranges: List[Tuple[int, int]] = []
    if n_frames < clip_len:
        return [(0, n_frames)]
        
    start = 0
    while start + clip_len <= n_frames:
        ranges.append((start, start + clip_len))
        start += hop
        
    # Ensure we get at least one clip even if it's shorter than clip_len
    if not ranges and n_frames > 0:
        return [(0, n_frames)]
        
    return ranges

class VideoProcessingError(Exception):
    """Raised when video processing encounters an error."""
    pass

def extract_for_video(
    video_path: Union[Path, str],
    out_root: Union[Path, str],
    fps: float,
    size_hw: Tuple[int, int],
    clip_seconds: float,
    hop_seconds: float,
    write_all_frames: bool = False,
    jpg_quality: int = 95,
    image_format: ImageFormat = DEFAULT_IMG_FORMAT,
) -> Dict[str, Any]:
    """Extract and process frames from a video file.
    
    Args:
        video_path: Path to input video file
        out_root: Root directory for output
        fps: Target frames per second (0 = use source FPS)
        size_hw: Target (height, width) for output frames
        clip_seconds: Duration of each clip in seconds
        hop_seconds: Time between clip starts in seconds
        write_all_frames: Whether to save all frames or just clip frames
        jpg_quality: JPEG quality (1-100) if using JPEG format
        image_format: Format to save images (JPG, PNG, WEBP)
        
    Returns:
        Dictionary with processing metadata and statistics
        
    Raises:
        VideoProcessingError: If video processing fails
        FileNotFoundError: If video file doesn't exist
        ValueError: For invalid parameters
        OSError: For filesystem errors
    """
    video_path = Path(video_path)
    out_root = Path(out_root)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not video_path.is_file():
        raise ValueError(f"Path is not a file: {video_path}")
    if size_hw[0] <= 0 or size_hw[1] <= 0:
        raise ValueError(f"Invalid frame dimensions: {size_hw}")
    if clip_seconds <= 0:
        raise ValueError(f"Clip seconds must be positive, got {clip_seconds}")
    if hop_seconds <= 0:
        raise ValueError(f"Hop seconds must be positive, got {hop_seconds}")
    if jpg_quality < 1 or jpg_quality > 100:
        raise ValueError(f"JPEG quality must be between 1-100, got {jpg_quality}")
    
    # Initialize video capture
    vcap = cv2.VideoCapture(str(video_path))
    if not vcap.isOpened():
        raise VideoProcessingError(f"Could not open video: {video_path}")
    
    try:
        # Get video properties
        src_fps = vcap.get(cv2.CAP_PROP_FPS)
        if src_fps <= 0:
            src_fps = 30.0  # Default FPS if not available
            
        src_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if src_frames <= 0:
            raise VideoProcessingError(f"Could not determine number of frames in {video_path}")
            
        # Calculate target FPS and frame indices
        tgt_fps = fps if fps > 0 else src_fps
        keep_idxs = set(sample_indices(src_frames, src_fps, tgt_fps))
        
        # Create output directories
        name = video_path.stem
        out_dir = out_root / name
        frames_dir = out_dir / "frames"
        clips_dir = out_dir / "clips"
        
        ensure_dir(out_dir)
        if write_all_frames:
            ensure_dir(frames_dir)
        ensure_dir(clips_dir)
        
        # Set up image writing parameters
        imwrite_params = []
        if image_format in (ImageFormat.JPG, ImageFormat.WEBP):
            imwrite_params = [
                int(cv2.IMWRITE_JPEG_QUALITY if image_format == ImageFormat.JPG 
                    else cv2.IMWRITE_WEBP_QUALITY),
                jpg_quality
            ]
            
        # Process frames
        frame_paths: List[Path] = []
        frame_count = 0
        saved_count = 0
        
        with tqdm(total=src_frames, desc=f"[video] {name}", unit="frame") as pbar:
            while True:
                ret, frame = vcap.read()
                if not ret:
                    break
                    
                if frame_count in keep_idxs:
                    try:
                        # Convert BGR to RGB and process
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        processed = center_resize_crop(
                            frame_rgb, 
                            size_hw[0], 
                            size_hw[1]
                        )
                        
                        # Save frame
                        frame_name = f"frame_{saved_count:06d}{image_format.value}"
                        if write_all_frames:
                            frame_path = frames_dir / frame_name
                        else:
                            frame_path = out_dir / f"tmp_{frame_name}"
                            
                        # Save with appropriate format
                        if image_format == ImageFormat.PNG:
                            success = cv2.imwrite(
                                str(frame_path),
                                cv2.cvtColor(processed, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_PNG_COMPRESSION, 9]  # Max compression
                            )
                        else:
                            success = cv2.imwrite(
                                str(frame_path),
                                cv2.cvtColor(processed, cv2.COLOR_RGB2BGR),
                                imwrite_params
                            )
                            
                        if not success:
                            raise VideoProcessingError(f"Failed to write frame {frame_path}")
                            
                        frame_paths.append(frame_path)
                        saved_count += 1
                        
                    except Exception as e:
                        raise VideoProcessingError(
                            f"Error processing frame {frame_count}: {str(e)}"
                        ) from e
                        
                frame_count += 1
                pbar.update(1)
                
        # If no frames were saved, we're done
        if not frame_paths:
            return {
                "status": "success",
                "message": "No frames were processed",
                "video_path": str(video_path),
                "frames_processed": 0,
                "clips_created": 0
            }
            
        # Create clips
        clip_len = int(round(tgt_fps * clip_seconds))
        hop = int(round(tgt_fps * hop_seconds))
        clip_ranges = chunk_ranges(len(frame_paths), clip_len, hop)
        
        # Prepare manifest
        manifest = {
            "video_name": name,
            "source_path": str(video_path),
            "resolution": size_hw,
            "source_fps": float(src_fps),
            "target_fps": float(tgt_fps),
            "clip_seconds": float(clip_seconds),
            "hop_seconds": float(hop_seconds),
            "total_frames": src_frames,
            "processed_frames": len(frame_paths),
            "clips": []
        }
        
        # Process clips
        for clip_idx, (start, end) in enumerate(clip_ranges):
            clip_dir = clips_dir / f"clip_{clip_idx:04d}"
            ensure_dir(clip_dir)
            
            # Copy frames to clip directory
            for i in range(start, min(end, len(frame_paths))):
                src = frame_paths[i]
                dst = clip_dir / f"frame_{i-start:06d}{image_format.value}"
                link_or_copy(src, dst)
            
            # Add to manifest
            manifest["clips"].append({
                "clip_idx": clip_idx,
                "start_frame": start,
                "end_frame": min(end, len(frame_paths)),
                "num_frames": min(end, len(frame_paths)) - start,
                "start_sec": start / tgt_fps,
                "end_sec": min(end, len(frame_paths)) / tgt_fps,
                "frames_dir": str(clip_dir)
            })
        
        # Save manifest
        manifest_path = out_dir / "clips.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Clean up temporary files if not keeping all frames
        if not write_all_frames:
            for path in frame_paths:
                try:
                    path.unlink()
                except OSError as e:
                    print(f"Warning: Could not remove temporary file {path}: {e}", 
                          file=sys.stderr)
        
        return {
            "status": "success",
            "manifest_path": str(manifest_path),
            "video_path": str(video_path),
            "frames_processed": len(frame_paths),
            "clips_created": len(clip_ranges),
            "output_dir": str(out_dir)
        }
        
    except Exception as e:
        raise VideoProcessingError(
            f"Error processing video {video_path}: {str(e)}"
        ) from e
        
    finally:
        # Always release video capture
        vcap.release()

def main() -> None:
    """Main entry point for frame extraction script."""
    parser = argparse.ArgumentParser(description="Extract frames from videos and organize into clips.")
    parser.add_argument(
        "--input", 
        type=Path, 
        required=True,
        help="Input video file or directory containing videos"
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        required=True,
        help="Output directory for processed frames and clips"
    )
    parser.add_argument(
        "--fps", 
        type=float, 
        default=0,
        help="Target frames per second (0 = use source FPS)"
    )
    parser.add_argument(
        "--size",
        type=str,
        default="256x256",
        help="Output frame size as 'HxW' or single number for square"
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=3.0,
        help="Duration of each clip in seconds"
    )
    parser.add_argument(
        "--hop-seconds",
        type=float,
        default=1.0,
        help="Time between clip starts in seconds"
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep all extracted frames (not just clips)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=[f.value for f in ImageFormat],
        default=DEFAULT_IMG_FORMAT.value,
        help="Output image format"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG/WEBP quality (1-100)"
    )
    
    args = parser.parse_args()
    
    try:
        # Parse size
        size_hw = parse_hw(args.size)
        
        # Get video files
        input_path = args.input
        if input_path.is_file():
            video_files = [input_path]
        else:
            video_files = list(input_path.glob("**/*.mp4")) + \
                         list(input_path.glob("**/*.avi")) + \
                         list(input_path.glob("**/*.mov")) + \
                         list(input_path.glob("**/*.mkv"))
        
        if not video_files:
            print(f"No video files found in {input_path}", file=sys.stderr)
            return
        
        # Process each video
        for video_file in video_files:
            try:
                print(f"\nProcessing {video_file.name}...")
                result = extract_for_video(
                    video_path=video_file,
                    out_root=args.output,
                    fps=args.fps,
                    size_hw=size_hw,
                    clip_seconds=args.clip_seconds,
                    hop_seconds=args.hop_seconds,
                    write_all_frames=args.keep_frames,
                    jpg_quality=args.quality,
                    image_format=ImageFormat(args.format)
                )
                print(f"Processed {result['frames_processed']} frames into {result['clips_created']} clips")
                print(f"Output saved to {result['output_dir']}")
                
            except Exception as e:
                print(f"Error processing {video_file}: {str(e)}", file=sys.stderr)
                if isinstance(e, VideoProcessingError) and hasattr(e, '__cause__') and e.__cause__:
                    print(f"  Caused by: {str(e.__cause__)}", file=sys.stderr)
                continue
    
    except Exception as e:
        print(f"Fatal error: {str(e)}", file=sys.stderr)
        if hasattr(e, '__cause__') and e.__cause__:
            print(f"  Caused by: {str(e.__cause__)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
