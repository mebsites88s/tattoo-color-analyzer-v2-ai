"""
Tattoo Color Analyzer

AI-powered tattoo analysis for laser removal difficulty assessment.
Developed by Think Again Tattoo Removal.

https://thinkagaintattooremoval.com

Features:
- Deep learning tattoo segmentation (SAM)
- Neural network session prediction
- Calibrated confidence scoring
- Color extraction and classification
- Clinical difficulty scoring
"""

__version__ = "2.0.0"
__author__ = "Think Again Tattoo Removal"

# Core modules (always available)
from .analyzer import TattooAnalyzer
from .color_classifier import ColorClassifier
from .difficulty_scorer import DifficultyScorer

# AI modules (optional, require torch)
try:
    from .segmentation import TattooSegmenter, segment_tattoo
    AI_SEGMENTATION = True
except ImportError:
    AI_SEGMENTATION = False

try:
    from .session_predictor import SessionPredictor, predict_sessions
    AI_PREDICTION = True
except ImportError:
    AI_PREDICTION = False

try:
    from .confidence import ConfidenceScorer, ConfidenceScore, compute_image_quality_metrics
    AI_CONFIDENCE = True
except ImportError:
    AI_CONFIDENCE = False

# Feature flags
AI_FEATURES = {
    "segmentation": AI_SEGMENTATION,
    "prediction": AI_PREDICTION,
    "confidence": AI_CONFIDENCE,
}

__all__ = [
    "TattooAnalyzer",
    "ColorClassifier", 
    "DifficultyScorer",
    "TattooSegmenter",
    "SessionPredictor",
    "ConfidenceScorer",
    "AI_FEATURES",
]
