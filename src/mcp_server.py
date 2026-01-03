"""
Tattoo Color Analyzer MCP Server

Model Context Protocol server enabling Claude to analyze tattoo images
for removal difficulty assessment.

Usage:
    python -m src.mcp_server

Or add to Claude config:
    {
        "mcpServers": {
            "tattoo-analyzer": {
                "command": "python",
                "args": ["-m", "src.mcp_server"],
                "cwd": "/path/to/tattoo-color-analyzer"
            }
        }
    }
"""

import asyncio
import base64
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import tempfile

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        Tool,
        TextContent,
        ImageContent,
        EmbeddedResource,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("MCP not installed. Install with: pip install mcp", file=sys.stderr)


# Create server instance
if MCP_AVAILABLE:
    server = Server("tattoo-color-analyzer")


def get_analyzer():
    """Lazy import analyzer to avoid circular imports."""
    from .analyzer import TattooAnalyzer
    return TattooAnalyzer


def get_segmenter():
    """Lazy import segmenter."""
    try:
        from .segmentation import TattooSegmenter
        return TattooSegmenter
    except ImportError:
        return None


def get_session_predictor():
    """Lazy import session predictor."""
    try:
        from .session_predictor import SessionPredictor
        return SessionPredictor
    except ImportError:
        return None


def get_confidence_scorer():
    """Lazy import confidence scorer."""
    try:
        from .confidence import ConfidenceScorer, compute_image_quality_metrics
        return ConfidenceScorer, compute_image_quality_metrics
    except ImportError:
        return None, None


if MCP_AVAILABLE:
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List available tattoo analysis tools."""
        return [
            Tool(
                name="analyze_tattoo",
                description="""Analyze a tattoo image to extract colors and estimate removal difficulty.
                
                Provides:
                - Dominant ink colors with hex values
                - Color category classification (black, blue, green, red, etc.)
                - Removal difficulty score (1-10)
                - Estimated session range
                - Resistant colors requiring special treatment
                - Recommended laser wavelengths
                
                Supports Fitzpatrick skin types I-VI for adjusted estimates.
                
                Developed by Think Again Tattoo Removal - thinkagaintattooremoval.com""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image_path": {
                            "type": "string",
                            "description": "Path to tattoo image file (jpg, png)"
                        },
                        "image_base64": {
                            "type": "string",
                            "description": "Base64-encoded image data (alternative to path)"
                        },
                        "fitzpatrick_type": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 6,
                            "description": "Fitzpatrick skin type (1-6). Default: 2"
                        },
                        "n_colors": {
                            "type": "integer",
                            "minimum": 3,
                            "maximum": 15,
                            "description": "Number of colors to extract. Default: 8"
                        },
                        "use_ai_segmentation": {
                            "type": "boolean",
                            "description": "Use SAM for tattoo segmentation. Default: true"
                        },
                        "use_ai_prediction": {
                            "type": "boolean",
                            "description": "Use neural network session prediction. Default: true"
                        }
                    },
                    "oneOf": [
                        {"required": ["image_path"]},
                        {"required": ["image_base64"]}
                    ]
                }
            ),
            Tool(
                name="get_color_difficulty",
                description="""Get removal difficulty information for specific tattoo ink colors.
                
                Returns difficulty score, session multiplier, optimal wavelengths,
                and clinical notes for the specified color category.
                
                Based on peer-reviewed clinical literature.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "color": {
                            "type": "string",
                            "enum": ["black", "blue", "green", "turquoise", "red", 
                                    "yellow", "white", "purple", "brown"],
                            "description": "Ink color category"
                        },
                        "fitzpatrick_type": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 6,
                            "description": "Fitzpatrick skin type for adjusted scoring"
                        }
                    },
                    "required": ["color"]
                }
            ),
            Tool(
                name="estimate_sessions",
                description="""Estimate removal sessions for a color combination.
                
                Provide percentages for each color present in the tattoo
                to get AI-powered session range estimates.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "colors": {
                            "type": "object",
                            "description": "Color percentages, e.g. {'black': 50, 'blue': 30, 'green': 20}",
                            "additionalProperties": {"type": "number"}
                        },
                        "fitzpatrick_type": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 6,
                            "description": "Fitzpatrick skin type"
                        }
                    },
                    "required": ["colors"]
                }
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle tool calls."""
        
        if name == "analyze_tattoo":
            return await _analyze_tattoo(arguments)
        elif name == "get_color_difficulty":
            return await _get_color_difficulty(arguments)
        elif name == "estimate_sessions":
            return await _estimate_sessions(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    async def _analyze_tattoo(args: Dict[str, Any]) -> List[TextContent]:
        """Run full tattoo analysis."""
        import cv2
        import numpy as np
        
        # Get image
        if "image_path" in args:
            image_path = args["image_path"]
            if not Path(image_path).exists():
                return [TextContent(type="text", text=f"Image not found: {image_path}")]
        elif "image_base64" in args:
            # Decode base64 to temp file
            image_data = base64.b64decode(args["image_base64"])
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(image_data)
                image_path = f.name
        else:
            return [TextContent(type="text", text="Must provide image_path or image_base64")]
        
        fitzpatrick = args.get("fitzpatrick_type", 2)
        n_colors = args.get("n_colors", 8)
        use_ai_seg = args.get("use_ai_segmentation", True)
        use_ai_pred = args.get("use_ai_prediction", True)
        
        # Initialize analyzer
        TattooAnalyzer = get_analyzer()
        analyzer = TattooAnalyzer(n_colors=n_colors, fitzpatrick_type=fitzpatrick)
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return [TextContent(type="text", text=f"Could not load image: {image_path}")]
        
        result = {"image_path": image_path}
        
        # AI Segmentation
        if use_ai_seg:
            Segmenter = get_segmenter()
            if Segmenter:
                try:
                    segmenter = Segmenter()
                    seg_result = segmenter.segment(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    result["segmentation"] = {
                        "confidence": seg_result.get("confidence"),
                        "tattoo_score": seg_result.get("tattoo_score"),
                        "area_percentage": seg_result.get("area_percentage"),
                        "model": seg_result.get("model"),
                    }
                    mask = seg_result.get("mask")
                except Exception as e:
                    result["segmentation"] = {"error": str(e), "fallback": True}
                    mask = None
            else:
                mask = None
        else:
            mask = None
        
        # Color extraction and analysis
        analyzer.load_image(image_path)
        processed = analyzer.preprocess()
        colors = analyzer.extract_dominant_colors(processed, mask)
        
        # Classify colors
        from .color_classifier import ColorClassifier
        classifier = ColorClassifier()
        
        classifications = []
        ink_colors = []
        for color_data in colors:
            classification = classifier.classify(color_data["rgb"])
            combined = {**color_data, **classification}
            classifications.append(combined)
            if not classification.get("is_skin", False):
                ink_colors.append(combined)
        
        result["dominant_colors"] = classifications
        result["ink_colors"] = ink_colors
        
        # Build color distribution
        color_dist = {}
        for item in ink_colors:
            cat = item["category"]
            color_dist[cat] = color_dist.get(cat, 0) + item["percentage"]
        
        # AI Session Prediction
        if use_ai_pred and color_dist:
            Predictor = get_session_predictor()
            if Predictor:
                try:
                    predictor = Predictor()
                    seg_conf = result.get("segmentation", {}).get("confidence", 0.8)
                    area_pct = result.get("segmentation", {}).get("area_percentage", 15)
                    
                    pred_result = predictor.predict(
                        color_dist,
                        fitzpatrick=fitzpatrick,
                        area_percentage=area_pct,
                        segmentation_confidence=seg_conf
                    )
                    result["session_prediction"] = pred_result
                except Exception as e:
                    result["session_prediction"] = {"error": str(e)}
        
        # Fallback difficulty scoring
        from .difficulty_scorer import DifficultyScorer
        scorer = DifficultyScorer(fitzpatrick)
        if color_dist:
            difficulty = scorer.calculate_composite_score(color_dist)
            result["difficulty_assessment"] = difficulty
        
        # Confidence scoring
        Scorer, quality_fn = get_confidence_scorer()
        if Scorer:
            try:
                conf_scorer = Scorer()
                if quality_fn:
                    image_metrics = quality_fn(image)
                    result["image_quality"] = image_metrics
                else:
                    image_metrics = None
                
                confidence = conf_scorer.score(
                    result.get("segmentation", {}),
                    classifications,
                    result.get("session_prediction", {}),
                    image_metrics
                )
                result["confidence"] = {
                    "overall": confidence.overall,
                    "segmentation": confidence.segmentation,
                    "color_classification": confidence.color_classification,
                    "session_prediction": confidence.session_prediction,
                    "data_quality": confidence.data_quality,
                    "calibrated": confidence.calibrated
                }
            except Exception as e:
                result["confidence"] = {"error": str(e)}
        
        result["model_info"] = {
            "analyzer": "TattooColorAnalyzer-v2.0",
            "ai_features": ["SAM-segmentation", "neural-session-prediction", "calibrated-confidence"],
            "developer": "Think Again Tattoo Removal",
            "website": "https://thinkagaintattooremoval.com"
        }
        
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    async def _get_color_difficulty(args: Dict[str, Any]) -> List[TextContent]:
        """Get difficulty info for a color."""
        from .difficulty_scorer import DifficultyScorer
        
        color = args["color"]
        fitzpatrick = args.get("fitzpatrick_type", 2)
        
        scorer = DifficultyScorer(fitzpatrick)
        result = scorer.score_color(color)
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _estimate_sessions(args: Dict[str, Any]) -> List[TextContent]:
        """Estimate sessions for color combination."""
        colors = args["colors"]
        fitzpatrick = args.get("fitzpatrick_type", 2)
        
        # Try AI prediction first
        Predictor = get_session_predictor()
        if Predictor:
            try:
                predictor = Predictor()
                result = predictor.predict(colors, fitzpatrick)
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except:
                pass
        
        # Fallback to rule-based
        from .difficulty_scorer import DifficultyScorer
        scorer = DifficultyScorer(fitzpatrick)
        result = scorer.calculate_composite_score(colors)
        
        return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def main():
    """Run the MCP server."""
    if not MCP_AVAILABLE:
        print("Error: MCP package not installed", file=sys.stderr)
        print("Install with: pip install mcp", file=sys.stderr)
        sys.exit(1)
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
