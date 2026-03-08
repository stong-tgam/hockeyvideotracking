import cv2
import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict

class HockeyRinkHomography:
    def __init__(self):
        """
        Initialize the homography calculator for hockey rink perspective transformation
        """
        # Define standard NHL hockey rink dimensions in feet
        # Length: 200 feet, Width: 85 feet
        self.rink_width_feet = 85
        self.rink_length_feet = 200

        # Define key points on a standard hockey rink (in feet from center)
        # Format: (x, y) in feet from center of rink
        self.rink_keypoints = {
            # Center ice
            'center_ice': (0, 0),

            # Faceoff dots (8 total: 4 face-off spots × 2 teams)
            'faceoff_center_left': (-20, 22),
            'faceoff_center_right': (-20, -22),
            'faceoff_left_zone_left': (20, 22),
            'faceoff_left_zone_right': (20, -22),
            'faceoff_right_zone_left': (-69, 22),  # Roughly positioned
            'faceoff_right_zone_right': (-69, -22),

            # Goal frames (centers of goals)
            'goal_left': (89, 0),  # 89 feet from center is approx the goal line
            'goal_right': (-89, 0),

            # Blue lines
            'blue_line_left': (64, 0),  # Blue lines are 64 feet from goal line
            'blue_line_right': (-64, 0),

            # Zone faceoff circles (approximate)
            'zone_circle_left': (79, 22),
            'zone_circle_right': (-79, 22),
            'zone_circle_left_bottom': (79, -22),
            'zone_circle_right_bottom': (-79, -22),
        }

        # Convert to meters (for real-world measurements)
        self.feet_to_meters = 0.3048

        # Initialize homography matrix
        self.homography_matrix = None
        self.is_initialized = False

        # Store correspondences for robust computation
        self.matched_points = {}

    def convert_rink_coords_to_pixels(self, pixel_width=800, pixel_height=400):
        """
        Convert rink coordinates (in feet from center) to pixel coordinates
        Maps rink (-89 to 89 feet in length, -42.5 to 42.5 feet in width) to pixels
        """
        # Map rink coordinates to pixel coordinates
        # Y is inverted because image Y increases downward
        pixel_coords = {}

        for key, (x_feet, y_feet) in self.rink_keypoints.items():
            # Scale and translate coordinates
            # Map [-89, 89] feet to [0, pixel_width] pixels
            # Map [-42.5, 42.5] feet to [pixel_height, 0] pixels (Y inverted)

            x_pixel = int(((x_feet + 89) / 178) * pixel_width)
            y_pixel = int(pixel_height - ((y_feet + 42.5) / 85) * pixel_height)

            pixel_coords[key] = (x_pixel, y_pixel)

        return pixel_coords

    def find_matching_keypoints(self, frame_detections):
        """
        Match detected objects in the frame to known rink keypoints
        frame_detections: Supervision detections object
        """
        matched_pairs_img = []  # Points in the image
        matched_pairs_rink = []  # Corresponding points in rink coordinates

        # Dictionary to hold detected points by class
        detected_centers = defaultdict(list)

        # Check if this is a hockey-specific model
        if hasattr(frame_detections, 'class_id') and len(frame_detections.class_id) > 0:
            for xyxy, class_id in zip(frame_detections.xyxy, frame_detections.class_id):
                center_x = int((xyxy[0] + xyxy[2]) / 2)
                center_y = int((xyxy[1] + xyxy[3]) / 2)

                # Map class IDs to rink features based on hockey model (adjust these as needed)
                # Assumed: 0: Center Ice, 1: Faceoff Dots, 2: Goal Frame, 3: Goaltender, 4: Players, 5: Puck, 6: Referee
                if class_id == 0:  # Center Ice
                    detected_centers['center_ice'].append((center_x, center_y))
                elif class_id == 1:  # Faceoff Dots (all of them)
                    detected_centers['faceoff_dots'].append((center_x, center_y))
                elif class_id == 2:  # Goal Frame
                    detected_centers['goal_frames'].append((center_x, center_y))

            # Match detected center ice to known center
            if 'center_ice' in detected_centers and len(detected_centers['center_ice']) > 0:
                # Take the first detected center ice (could improve with verification)
                img_point = detected_centers['center_ice'][0]
                rink_point = self.rink_keypoints['center_ice']
                matched_pairs_img.append(img_point)
                matched_pairs_rink.append(rink_point)

            # Match faceoff dots - need to match multiple
            if 'faceoff_dots' in detected_centers and len(detected_centers['faceoff_dots']) >= 2:
                faceoff_img_points = detected_centers['faceoff_dots']

                # Get corresponding rink coordinates for faceoff dots
                faceoff_rink_points = [self.rink_keypoints[k] for k in self.rink_keypoints.keys()
                                      if 'faceoff' in k][:len(faceoff_img_points)]

                if len(faceoff_rink_points) >= 2:
                    # Simple assignment based on proximity
                    for i, img_pt in enumerate(faceoff_img_points[:len(faceoff_rink_points)]):
                        matched_pairs_img.append(img_pt)
                        matched_pairs_rink.append(faceoff_rink_points[i])

            # Match goal frames
            if 'goal_frames' in detected_centers and len(detected_centers['goal_frames']) >= 2:
                goal_img_points = detected_centers['goal_frames']

                # Sort goal frames by x coordinate to distinguish left/right goals
                goal_img_points_sorted = sorted(goal_img_points, key=lambda x: x[0])

                if len(goal_img_points_sorted) >= 2:
                    # Left goal (smaller x) corresponds to right side of rink (positive x), and vice versa
                    # Actually, conventionally, left goal from home team perspective is negative x
                    matched_pairs_img.extend(goal_img_points_sorted[:2])
                    matched_pairs_rink.extend([self.rink_keypoints['goal_left'],
                                              self.rink_keypoints['goal_right']])

        # If we don't have enough specific hockey features, fall back to using all detections
        # as potential reference points for a rough estimate
        if len(matched_pairs_img) < 4:
            # Use all detected objects as potential reference points if we have too few
            all_img_points = [(int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2))
                             for xyxy in frame_detections.xyxy]

            # If we have at least 4 points, use them as rough references
            if len(all_img_points) >= 4:
                # Select 4 points: center, left, right, and a fourth as reference
                if len(all_img_points) >= 4:
                    matched_pairs_img = all_img_points[:4]

                    # Map to rough rink positions (this is a fallback)
                    # This assumes the video has a perspective view of the rink
                    matched_pairs_rink = [
                        self.rink_keypoints['center_ice'],
                        self.rink_keypoints['blue_line_left'],
                        self.rink_keypoints['blue_line_right'],
                        self.rink_keypoints['goal_left']
                    ][:len(matched_pairs_img)]

        return matched_pairs_img, matched_pairs_rink

    def compute_homography(self, frame_detections):
        """
        Compute homography matrix from detected keypoints
        """
        matched_img_pts, matched_rink_pts = self.find_matching_keypoints(frame_detections)

        if len(matched_img_pts) >= 4 and len(matched_rink_pts) >= 4:
            # Convert rink points from feet to pixels for computation
            # Get frame dimensions to normalize points for homography computation
            # In real implementation, you'd get the frame shape from the current frame
            frame_height, frame_width = 720, 1280  # Default assumed size, should be dynamic

            # We need to scale rink coordinates to match the relative proportions of the image
            # The rink is 200ft x 85ft, so we'll scale to fit in image dimensions proportionally
            scale_x = frame_width / self.rink_length_feet
            scale_y = frame_height / self.rink_width_feet
            scale = min(scale_x, scale_y)  # Maintain aspect ratio

            # Convert rink feet coordinates to image space
            scaled_rink_pts = []
            for x_feet, y_feet in matched_rink_pts:
                # Scale and shift to center
                x_scaled = int(x_feet * scale + frame_width // 2)
                y_scaled = int(frame_height // 2 - y_feet * scale)  # Y is inverted
                scaled_rink_pts.append([x_scaled, y_scaled])

            # Convert to numpy arrays
            img_pts = np.array(matched_img_pts, dtype=np.float32)
            rink_pts = np.array(scaled_rink_pts, dtype=np.float32)

            # Compute homography matrix using RANSAC for robustness
            try:
                homography_matrix, mask = cv2.findHomography(
                    srcPoints=rink_pts,
                    dstPoints=img_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=10.0  # Increased tolerance for initial estimation
                )

                if homography_matrix is not None:
                    self.homography_matrix = homography_matrix
                    self.is_initialized = True
                    return homography_matrix
                else:
                    # Fallback: return previous matrix or identity
                    return self.homography_matrix
            except cv2.error as e:
                print(f"OpenCV error computing homography: {e}")
                return self.homography_matrix
        else:
            # Not enough points to compute homography
            # Return previous matrix if available
            return self.homography_matrix

    def transform_point_to_rink_space(self, img_point, feet_units=True):
        """
        Transform a point from image space to rink space (feet or meters)
        img_point: (x, y) in image coordinates
        feet_units: if True, return in feet; if False, return in meters
        """
        if self.homography_matrix is None:
            return None

        # Homogeneous coordinates
        img_point_homo = np.array([[img_point]], dtype=np.float32)

        # Transform using inverse homography (image -> rink)
        # We need inverse because homography maps rink->image, we want image->rink
        if self.homography_matrix.shape == (3, 3):
            try:
                # Calculate inverse of homography matrix
                inv_h = np.linalg.inv(self.homography_matrix)
                rink_point_homo = cv2.perspectiveTransform(img_point_homo, inv_h)

                rink_point = (float(rink_point_homo[0][0][0]), float(rink_point_homo[0][0][1]))

                # Need to scale back from pixels to rink units
                # This is a simplification - in a real implementation,
                # you'd need to track the scaling applied during homography computation

                # For now, let's calculate based on typical rink to image ratios
                frame_width, frame_height = 1280, 720  # Typical frame size
                px_to_feet_x = self.rink_length_feet / frame_width
                px_to_feet_y = self.rink_width_feet / frame_height

                # Apply reverse scaling to get feet coordinates
                feet_x = (rink_point[0] - frame_width/2) * px_to_feet_x
                feet_y = -(rink_point[1] - frame_height/2) * px_to_feet_y  # Y inverted

                result = (feet_x, feet_y)

                if feet_units:
                    return result
                else:
                    # Convert to meters
                    return (result[0] * self.feet_to_meters,
                           result[1] * self.feet_to_meters)
            except np.linalg.LinAlgError:
                # Matrix is singular, return None
                return None
        return None

    def transform_point_from_rink_space(self, rink_point, feet_units=True):
        """
        Transform a point from rink space (feet or meters) to image space
        rink_point: (x, y) in rink coordinates
        feet_units: if True, rink_point is in feet; if False, in meters
        """
        if self.homography_matrix is None:
            return None

        # Convert to feet if needed
        if not feet_units:
            x_feet = rink_point[0] / self.feet_to_meters
            y_feet = rink_point[1] / self.feet_to_meters
            rink_point_scaled = (x_feet, y_feet)
        else:
            rink_point_scaled = rink_point

        # Convert rink coordinates to image coordinates based on frame size
        frame_width, frame_height = 1280, 720  # Typical frame size
        feet_to_px_x = frame_width / self.rink_length_feet
        feet_to_px_y = frame_height / self.rink_width_feet

        # Scale and shift to image coordinates
        x_img = rink_point_scaled[0] / feet_to_px_x + frame_width/2
        y_img = frame_height/2 - rink_point_scaled[1] / feet_to_px_y  # Y inverted

        # Homogeneous coordinates
        rink_point_homo = np.array([[[x_img, y_img]]], dtype=np.float32)

        # Transform using homography (this time using the forward transformation)
        img_point_homo = cv2.perspectiveTransform(rink_point_homo, self.homography_matrix)

        img_point = (int(img_point_homo[0][0][0]), int(img_point_homo[0][0][1]))
        return img_point

    def get_rink_boundary_points(self):
        """
        Get the four corners of the hockey rink in rink coordinates (feet)
        """
        half_length = self.rink_length_feet / 2
        half_width = self.rink_width_feet / 2

        # Corners: (x, y) in feet from center
        corners = [
            (-half_length, -half_width),  # Bottom-left
            (half_length, -half_width),   # Bottom-right
            (half_length, half_width),    # Top-right
            (-half_length, half_width)    # Top-left
        ]

        return corners

    def initialize_if_needed(self, frame_detections):
        """
        Initialize homography if not already initialized
        """
        if not self.is_initialized:
            self.compute_homography(frame_detections)