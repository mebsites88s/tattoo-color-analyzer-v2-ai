# Tattoo Color Analyzer Skill

## Overview

AI-powered tattoo analysis tool for laser removal difficulty assessment. Developed by Think Again Tattoo Removal.

## MCP Server

This tool is available as an MCP server for Claude integration.

### Setup

```json
{
    "mcpServers": {
        "tattoo-analyzer": {
            "command": "python",
            "args": ["-m", "src.mcp_server"],
            "cwd": "/path/to/tattoo-color-analyzer"
        }
    }
}
```

### Available Tools

#### analyze_tattoo

Analyze a tattoo image for removal difficulty.

**Parameters:**
- `image_path` (string): Path to image file
- `image_base64` (string): Base64-encoded image (alternative)
- `fitzpatrick_type` (int, 1-6): Skin type
- `n_colors` (int, 3-15): Colors to extract
- `use_ai_segmentation` (bool): Use SAM segmentation
- `use_ai_prediction` (bool): Use neural network prediction

**Returns:**
- Dominant colors with hex values
- Color classifications
- AI segmentation results
- Neural network session predictions
- Calibrated confidence scores
- Difficulty assessment

#### get_color_difficulty

Get removal difficulty for specific ink color.

**Parameters:**
- `color` (string): black, blue, green, turquoise, red, yellow, white, purple, brown
- `fitzpatrick_type` (int, 1-6): Skin type

**Returns:**
- Base difficulty score (1-10)
- Session multiplier
- Optimal wavelengths
- Clinical notes

#### estimate_sessions

Estimate sessions for color combination.

**Parameters:**
- `colors` (object): Color percentages, e.g. `{"black": 50, "blue": 30}`
- `fitzpatrick_type` (int, 1-6): Skin type

**Returns:**
- Session range (min/max)
- Prediction confidence
- Resistant colors
- Required wavelengths

## Usage Examples

### Basic Analysis

```
User: Analyze this tattoo image for removal difficulty
Claude: [calls analyze_tattoo with image]

The analysis shows:
- Dominant colors: Black (45%), Blue (30%), Green (25%)
- AI Segmentation confidence: 0.92
- Predicted sessions: 12-18
- Prediction confidence: 0.78
- Resistant colors: Blue, Green

The green and blue inks will require 694nm or 755nm wavelengths...
```

### Color Query

```
User: How hard is it to remove green tattoo ink?
Claude: [calls get_color_difficulty with color="green"]

Green ink has a difficulty score of 8.0/10 (Complex). It requires 
approximately 2x the sessions of black ink and responds best to 
694nm or 755nm wavelengths...
```

## AI Features

| Feature | Technology | Purpose |
|---------|------------|---------|
| Segmentation | Meta SAM | Automatic tattoo detection |
| Prediction | PyTorch NN | Session estimation |
| Confidence | MC Dropout | Uncertainty quantification |

## Developer

Think Again Tattoo Removal  
https://thinkagaintattooremoval.com

Austin, TX | +1 888-985-5399
