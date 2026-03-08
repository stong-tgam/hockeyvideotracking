# Hockey Rink Template and Keypoints Guide

## Standard NHL Hockey Rink Dimensions
- Length: 200 feet (61.0 m)
- Width: 85 feet (25.9 m)
- Corner radius: 28 feet (8.5 m)

## Key Points on Hockey Rink (measured from center point)
The following are the precise locations of features on a standard hockey rink:

### Primary Features (Used for Homography)
1. **Center Ice Dot**: (0, 0) feet
2. **Faceoff Dots**:
   - Center ice faceoff dots: (0, 22), (0, -22) feet
   - Left zone faceoff dots: (20, 22), (20, -22) feet
   - Right zone faceoff dots: (-69, 22), (-69, -22) feet
3. **Goal Frame Centers**:
   - Left goal: (89, 0) feet (right goal when facing rink from TV broadcast)
   - Right goal: (-89, 0) feet

### Additional Reference Points (for validation)
4. **Blue Lines**:
   - Left blue line: (64, 0) feet
   - Right blue line: (-64, 0) feet
5. **Goal Lines**:
   - Left goal line: (89, y) feet for any y
   - Right goal line: (-89, y) feet for any y

## Creating Your Rink Template Image

### Option 1: Use a Real Rink Photo
1. Find a bird's eye view photo of an actual hockey rink
2. Ensure the photo clearly shows:
   - Center ice circle and dot
   - All 9 faceoff dots (5 in neutral zone, 4 in attacking zones)
   - Both goal frames/goal lines
3. Crop to focus on the playing surface
4. Image should be high resolution and minimally distorted

### Option 2: Create a Schematic
1. Use graphic software to draw a rectangle (200x85 units)
2. Mark all key points mentioned above
3. Ensure accurate proportions and positions
4. Export as a high-resolution image

## Format for Homography Mapping

The homography module expects the following:

1. The algorithm automatically detects:
   - Center ice dot (class 0)
   - Faceoff dots (class 1)
   - Goal frames (class 2)

2. The algorithm matches these detected features to the known rink coordinates based on proximity and geometric relationships.

## Important Notes

1. **Camera Calibration**: Since your XbotGo camera pans and rotates, ensure there are sufficient static reference points (faceoff dots, goal frames, center ice) visible in each frame to compute accurate homography transformations.

2. **Accuracy Considerations**: The more precisely the detection model can locate the reference features, the more accurate the homography transformation will be.

3. **Fallback Strategy**: If insufficient reference points are detected in a frame, the system will use the last computed homography matrix.

## Sample Keypoints Dictionary Structure

```python
# This is already implemented in the homography.py module:
rink_keypoints = {
    # Center ice
    'center_ice': (0, 0),

    # Faceoff dots
    'faceoff_center_left': (-20, 22),
    'faceoff_center_right': (-20, -22),
    'faceoff_left_zone_left': (20, 22),
    'faceoff_left_zone_right': (20, -22),
    'faceoff_right_zone_left': (-69, 22),
    'faceoff_right_zone_right': (-69, -22),

    # Goal frames
    'goal_left': (89, 0),
    'goal_right': (-89, 0),

    # Additional reference points
    'blue_line_left': (64, 0),
    'blue_line_right': (-64, 0),
}
```

## Validation Process

To validate that your setup is working correctly:

1. Run the system on a sample video where the camera remains relatively stable
2. Verify that players and the puck are mapped correctly to the 2D rink view
3. Check that distance measurements are reasonable (players don't appear to move impossibly far between frames)
4. Confirm that puck possession logic behaves sensibly

The homography algorithm is designed to be robust to the dynamic camera movement of your XbotGo setup, continuously updating the perspective transformation as reference points are detected in each frame.