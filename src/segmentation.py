"""
Tattoo Segmentation Module

Uses Meta's Segment Anything Model (SAM) for automatic tattoo region detection.
This replaces the basic saturation-based masking with deep learning segmentation.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
import cv2

# Lazy imports for optional SAM dependency
_sam_model = None
_sam_predictor = None


def _load_sam_model(model_type: str = "vit_b", checkpoint_path: Optional[str] = None):
    """
    Lazy-load SAM model to avoid startup overhead.
    
    Args:
        model_type: SAM model variant ('vit_h', 'vit_l', 'vit_b')
        checkpoint_path: Path to model checkpoint. Downloads if None.
    
    Returns:
        SAM predictor instance
    """
    global _sam_model, _sam_predictor
    
    if _sam_predictor is not None:
        return _sam_predictor
    
    try:
        from segment_anything import sam_model_registry, SamPredictor
        import torch
    except ImportError:
        raise ImportError(
            "SAM not installed. Install with: pip install segment-anything\n"
            "Also requires: pip install torch torchvision"
        )
    
    if checkpoint_path is None:
        checkpoint_dir = Path.home() / ".cache" / "tattoo-analyzer" / "sam"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_map = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth",
        }
        checkpoint_path = checkpoint_dir / checkpoint_map[model_type]
        
        if not checkpoint_path.exists():
            print(f"Downloading SAM {model_type} checkpoint...")
            _download_checkpoint(model_type, checkpoint_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    _sam_model = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    _sam_model.to(device=device)
    _sam_predictor = SamPredictor(_sam_model)
    
    return _sam_predictor


def _download_checkpoint(model_type: str, save_path: Path):
    """Download SAM checkpoint from Meta."""
    import urllib.request
    
    urls = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }
    
    url = urls[model_type]
    urllib.request.urlretrieve(url, save_path)


class TattooSegmenter:
    """
    Deep learning-based tattoo segmentation using Segment Anything Model.
    
    Provides automatic detection and segmentation of tattooed regions
    without manual annotation or region selection.
    """
    
    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint_path: Optional[str] = None,
        use_auto_mask: bool = True
    ):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.use_auto_mask = use_auto_mask
        self._predictor = None
        self._mask_generator = None
    
    def _ensure_loaded(self):
        """Ensure model is loaded."""
        if self._predictor is None:
            self._predictor = _load_sam_model(self.model_type, self.checkpoint_path)
            
            if self.use_auto_mask:
                from segment_anything import SamAutomaticMaskGenerator
                self._mask_generator = SamAutomaticMaskGenerator(
                    _sam_model,
                    points_per_side=32,
                    pred_iou_thresh=0.86,
                    stability_score_thresh=0.92,
                    min_mask_region_area=100,
                )
    
    def segment(
        self,
        image: np.ndarray,
        point_prompts: Optional[List[Tuple[int, int]]] = None,
        return_all_masks: bool = False
    ) -> Dict[str, any]:
        """
        Segment tattoo regions from image.
        
        Args:
            image: Input image (RGB format)
            point_prompts: Optional (x, y) points indicating tattoo locations
            return_all_masks: Return all detected masks vs just best
        
        Returns:
            Dictionary with mask, confidence, bbox, area_percentage
        """
        self._ensure_loaded()
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if point_prompts is not None:
            return self._segment_with_points(image, point_prompts)
        elif self.use_auto_mask:
            return self._segment_automatic(image, return_all_masks)
        else:
            return self._segment_with_heuristic_points(image)
    
    def _segment_automatic(self, image: np.ndarray, return_all: bool = False) -> Dict:
        """Generate masks automatically and select best tattoo candidate."""
        masks = self._mask_generator.generate(image)
        
        if not masks:
            return self._fallback_segmentation(image)
        
        scored_masks = []
        for mask_data in masks:
            score = self._score_tattoo_likelihood(image, mask_data["segmentation"])
            scored_masks.append({**mask_data, "tattoo_score": score})
        
        scored_masks.sort(key=lambda x: x["tattoo_score"], reverse=True)
        
        if return_all:
            return {"masks": scored_masks}
        
        best = scored_masks[0]
        mask = best["segmentation"].astype(np.uint8) * 255
        
        return {
            "mask": mask,
            "confidence": float(best["predicted_iou"]),
            "tattoo_score": float(best["tattoo_score"]),
            "bbox": best["bbox"],
            "area_percentage": float(np.sum(mask > 0) / mask.size * 100),
            "model": f"SAM-{self.model_type}",
        }
    
    def _segment_with_points(self, image: np.ndarray, points: List[Tuple[int, int]]) -> Dict:
        """Segment using provided point prompts."""
        self._predictor.set_image(image)
        
        point_coords = np.array(points)
        point_labels = np.ones(len(points))
        
        masks, scores, _ = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        
        best_idx = np.argmax(scores)
        mask = masks[best_idx].astype(np.uint8) * 255
        
        coords = np.where(mask > 0)
        if len(coords[0]) > 0:
            bbox = [
                int(np.min(coords[1])), int(np.min(coords[0])),
                int(np.max(coords[1]) - np.min(coords[1])),
                int(np.max(coords[0]) - np.min(coords[0]))
            ]
        else:
            bbox = [0, 0, 0, 0]
        
        return {
            "mask": mask,
            "confidence": float(scores[best_idx]),
            "bbox": bbox,
            "area_percentage": float(np.sum(mask > 0) / mask.size * 100),
            "model": f"SAM-{self.model_type}",
        }
    
    def _score_tattoo_likelihood(self, image: np.ndarray, mask: np.ndarray) -> float:
        """Score how likely a mask represents a tattoo."""
        if np.sum(mask) == 0:
            return 0.0
        
        masked_pixels = image[mask]
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        masked_hsv = hsv[mask]
        
        sat_mean = np.mean(masked_hsv[:, 1]) / 255
        val_mean = 1 - (np.mean(masked_hsv[:, 2]) / 255)
        color_std = np.std(masked_pixels) / 255
        
        area_ratio = np.sum(mask) / mask.size
        size_score = max(0, min(1, 1 - abs(area_ratio - 0.15) * 2))
        
        return float(sat_mean * 0.3 + val_mean * 0.3 + color_std * 0.2 + size_score * 0.2)
    
    def _fallback_segmentation(self, image: np.ndarray) -> Dict:
        """Fallback to traditional segmentation if SAM fails."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        _, s, v = cv2.split(hsv)
        
        mask = np.zeros(s.shape, dtype=np.uint8)
        mask[(s > 30) | (v < 80)] = 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return {
            "mask": mask,
            "confidence": 0.5,
            "tattoo_score": 0.5,
            "bbox": [0, 0, image.shape[1], image.shape[0]],
            "area_percentage": float(np.sum(mask > 0) / mask.size * 100),
            "model": "fallback-hsv",
            "warning": "SAM segmentation failed, using fallback method"
        }
