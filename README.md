# Hockey Analytics System

A comprehensive computer vision system that analyzes hockey game footage to track players, puck, and generate spatial analytics on a 2D rink diagram. The system handles dynamic camera movements using homography transformation and implements robust tracking algorithms.

## Features

- Custom YOLOv8 model trained for hockey scenes (SimulaMet-HOST/HockeyAI/HockeyAI_model_weight.pt)
- Interactive player identification with jersey number assignment
- Robust tracking using BoT-SORT/ByteTrack algorithms
- Dynamic perspective transformation to map to 2D rink coordinates
- Puck interpolation during occlusions
- Real-world distance calculations
- Puck possession tracking
- KDE heatmaps for player movement analysis
- Side-by-side visualization of original video and 2D rink view

## Hardware Requirements

- NVIDIA RTX 3080 GPU (CUDA enabled)
- At least 16GB RAM (32GB recommended for high-resolution videos)
- Compatible with 720p+ video input

## Installation

1. Clone this repository
2. Install Python 3.8 or higher
3. Run setup to install dependencies and download the model:

```bash
python setup.py setup
```

This will:
- Install all required Python packages
- Download the HockeyAI model from Hugging Face

## Usage

### Basic Analysis

```bash
python setup.py run --video path/to/your/video.mp4
```

### With Output Video

```bash
python setup.py run --video path/to/your/video.mp4 --output output_video.mp4
```

### Using Custom Model

```bash
python setup.py run --video path/to/your/video.mp4 --model path/to/custom/model.pt
```

## Manual Player Initialization

On the first frame of your video, the system will pause and display bounding boxes around detected players. Click on each player and enter their jersey number when prompted. Press 'c' when you're done assigning numbers.

During playback, you can:
- Press 's' to select a player for heatmap visualization
- Press 'q' to quit the analysis

## Technical Architecture

### Modules

1. **Detector (`detector.py`)**: Runs YOLOv8 inference to detect all hockey objects
2. **Tracker (`tracker.py`)**: Maintains consistent object IDs using tracking algorithms
3. **Homography (`homography.py`)**: Computes perspective transformation between video and rink
4. **Analytics (`analytics.py`)**: Calculates metrics and generates heatmaps
5. **Visualization (`visualization.py`)**: Creates the final combined output
6. **Main (`main.py`)**: Orchestrates all components

### Object Classes

The HockeyAI model detects 7 classes:
- 0: Center Ice
- 1: Faceoff Dots
- 2: Goal Frame
- 3: Goaltender
- 4: Players
- 5: Puck
- 6: Referee

### Coordinate Systems

- **Image Space**: Pixel coordinates in the video frame
- **Rink Space**: Real-world coordinates in feet from center ice
- **Visualization Space**: Coordinates on the 2D rink diagram

## Key Algorithms

1. **Tracking**: BoT-SORT/ByteTrack for robust multi-object tracking
2. **Perspective**: Dynamic homography using CV2 with automatic reference point matching
3. **Interpolation**: Kalman filtering for puck position during occlusions
4. **Analytics**: Real-world distance calculation using calibrated homography
5. **Possession**: Distance-based logic with temporal consistency checks
6. **Heatmaps**: Kernel Density Estimation for player movement patterns

## Rink Template Mapping

The system automatically detects and matches:
- Center ice dot
- All faceoff dots (center and zone)
- Goal frame centers

These are matched to known positions on a standard NHL rink to compute the homography transformation. The system handles camera panning and zooming by continuously updating this transformation.

## Output Format

The system produces:
- A side-by-side video with:
  - Left: Original video with tracking overlays
  - Right: 2D rink diagram with player positions and puck tracking
- Real-time distance metrics for players
- Puck possession indicators
- KDE heatmaps for selected players

## Customization

You can modify the following parameters:
- Possession distance threshold in `analytics.py`
- Interpolation parameters in `tracker.py`
- Visualization colors and styles in `visualization.py`
- Tracking algorithm in `tracker.py`

## Troubleshooting

- If tracking is inconsistent, check that your video has sufficient lighting and the camera isn't shaking excessively
- If homography is incorrect, ensure the reference points (center ice, faceoff dots, goals) are clearly visible
- If players aren't being tracked correctly, try adjusting the confidence threshold in the detector

## Dependencies

- opencv-python
- ultralytics
- supervision
- numpy
- scipy
- torch
- matplotlib
- seaborn
- huggingface_hub
- filterpy

## Usage

### Download Model
```bash
python download_model.py
```

### Automated Analysis (Recommended)
```bash
# With output video file
python demo.py --video path/to/your/video.mp4 --model models/HockeyAI_model_weight.pt --output output_video.mp4

# Without saving output
python demo.py --video path/to/your/video.mp4 --model models/HockeyAI_model_weight.pt
```

### Interactive Mode
```bash
python main.py --video path/to/your/video.mp4 --model models/HockeyAI_model_weight.pt --output output_video.mp4
```

Controls during interactive mode:
- Press 'q' to quit
- Press 's' to select player for heatmap visualization

## Key Improvements

- **Automated Jersey Assignment**: The demo version automatically assigns jersey numbers based on tracker IDs, eliminating the need for manual input
- **Enhanced Video Output**: Improved codec compatibility with fallback options (MP4V, XVID, MJPG, X264) to ensure video files can be opened
- **Better Visualization**: Improved side-by-side view with proper scaling of the rink visualization
- **Robust Tracking**: Enhanced puck interpolation during occlusions
- **Failsafe Models**: Automatic fallback to general YOLO models if hockey-specific models fail to download

## Performance Notes

- The system is optimized for NVIDIA GPUs with CUDA
- Processing speed depends on video resolution and complexity
- Higher resolution videos will take longer to process but may provide better detection accuracy
- Consider processing videos in segments if working with very long recordings