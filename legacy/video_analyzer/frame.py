from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class Frame:
    number: int
    path: Path
    timestamp: float
    score: float

class VideoProcessor:
    # Class constants
    FRAME_DIFFERENCE_THRESHOLD = 10.0
    
    def __init__(self, video_path: Path, output_dir: Path, model: str):
        self.video_path = video_path
        self.output_dir = output_dir
        self.model = model
        self.frames: List[Frame] = []
        
    def _calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate the difference between two frames using absolute difference."""
        if frame1 is None or frame2 is None:
            return 0.0
        
        # Convert to grayscale for simpler comparison
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference and mean
        diff = cv2.absdiff(gray1, gray2)
        score = np.mean(diff)
        
        return float(score)

    def _is_keyframe(self, current_frame: np.ndarray, prev_frame: np.ndarray, threshold: float = FRAME_DIFFERENCE_THRESHOLD) -> bool:
        """Determine if frame is significantly different from previous frame."""
        if prev_frame is None:
            return True
            
        score = self._calculate_frame_difference(current_frame, prev_frame)
        return score > threshold

    def extract_keyframes(self, frames_per_minute: int = 10, duration: Optional[float] = None, max_frames: Optional[int] = None) -> List[Frame]:
        """Extract keyframes from video targeting a specific number of frames per minute."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps
        
        if duration:
            video_duration = min(duration, video_duration)
            total_frames = int(min(total_frames, duration * fps))
        
        # Calculate target number of frames
        target_frames = max(1, min(
            int((video_duration / 60) * frames_per_minute),
            total_frames,
            max_frames if max_frames is not None else float('inf')
        ))
        
        # Calculate adaptive sampling interval
        sample_interval = max(1, total_frames // (target_frames * 2))
        
        frame_candidates = []
        prev_frame = None
        frame_count = 0
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_interval == 0:
                score = self._calculate_frame_difference(frame, prev_frame)
                if score > self.FRAME_DIFFERENCE_THRESHOLD:
                    frame_candidates.append((frame_count, frame, score))
                prev_frame = frame.copy()
                
            frame_count += 1
            
        cap.release()
        
        # Select the most significant frames
        selected_candidates = sorted(frame_candidates, key=lambda x: x[2], reverse=True)[:target_frames]
        
        # If max_frames is specified, sample evenly across the candidates
        if max_frames is not None and max_frames < len(selected_candidates):
            step = len(selected_candidates) / max_frames
            selected_frames = [selected_candidates[int(i * step)] for i in range(max_frames)]
        else:
            selected_frames = selected_candidates

        self.frames = []
        for idx, (frame_num, frame, score) in enumerate(selected_frames):
            frame_path = self.output_dir / f"frame_{idx}.jpg"
            cv2.imwrite(str(frame_path), frame)
            timestamp = frame_num / fps
            self.frames.append(Frame(idx, frame_path, timestamp, score))
        
        logger.info(f"Extracted {len(self.frames)} frames from video (target was {target_frames})")
        return self.frames

    def extract_smart_keyframes(self, min_scene_len: int = 15) -> List[Frame]:
        """
        Use PySceneDetect for smart scene segmentation and extract the best frame from each scene.
        More accurate than simple pixel difference, captures true narrative changes.
        """
        try:
            from scenedetect import detect, AdaptiveDetector
        except ImportError:
            logger.error("scenedetect not installed. Please run 'pip install scenedetect'. Falling back to normal extraction.")
            return self.extract_keyframes()

        logger.info("Using smart scene detection algorithm...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use AdaptiveDetector to adapt to video content tempo
        scene_list = detect(str(self.video_path), AdaptiveDetector(min_scene_len=min_scene_len))
        
        extracted_frames = []
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        for i, scene in enumerate(scene_list):
            # Get the middle frame of the scene (usually most stable)
            start_frame, end_frame = scene
            middle_frame_idx = (start_frame.get_frames() + end_frame.get_frames()) // 2
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
            ret, frame_data = cap.read()
            
            if ret:
                # Blur detection: if middle frame is blurry, search forward for clear frame
                if self._is_blurry(frame_data):
                    # Simplified de-blur logic: try looking ahead up to 5 frames
                    for _ in range(5):
                        cap.read() # Read next frame
                        ret_next, frame_next = cap.read()
                        if ret_next and not self._is_blurry(frame_next):
                            frame_data = frame_next
                            middle_frame_idx += (_ + 1)
                            break

                timestamp = middle_frame_idx / fps
                frame_filename = self.output_dir / f"scene_{i:03d}_{timestamp:.2f}s.jpg"
                cv2.imwrite(str(frame_filename), frame_data)
                
                # Use a high score for scene changes
                extracted_frames.append(Frame(number=i, path=frame_filename, timestamp=timestamp, score=100.0))
        
        cap.release()
        if not extracted_frames:
             logger.warning("Smart detection found 0 scenes, falling back to standard extraction.")
             return self.extract_keyframes()
             
        logger.info(f"Smart detected {len(extracted_frames)} scene keyframes")
        self.frames = extracted_frames
        return extracted_frames

    def _is_blurry(self, image: np.ndarray, threshold: float = 100.0) -> bool:
        """Use Laplacian variance to detect blur."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(gray, cv2.CV_64F).var()
        return fm < threshold
