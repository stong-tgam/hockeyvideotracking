import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from collections import defaultdict, deque
import math
import csv
import os

class HockeyAnalytics:
    def __init__(self, homography_calculator):
        """
        Initialize analytics engine with homography calculator
        """
        self.homography_calc = homography_calculator
        self.player_distances = defaultdict(float)  # Total distance traveled by each player
        self.player_positions = {}  # Current positions of players in rink space
        self.puck_control_history = {}  # Track puck possession
        self.possession_threshold = 5.0  # Distance threshold in feet for puck control
        self.possession_min_frames = 5  # Minimum frames to establish possession

        # KDE heatmaps for players
        self.player_heatmaps = {}
        self.heatmap_resolution = (800, 400)  # Resolution for heatmap generation
        self.position_history = defaultdict(lambda: deque(maxlen=500))  # Store position history for KDE

        # CSV export functionality
        self.csv_data = []
        self.prev_positions = {}  # Keep track of previous positions for distance calculation

    def export_csv_data(self, output_path="player_analytics.csv"):
        """
        Export collected analytics data to CSV file
        """
        if not self.csv_data:
            print("No data to export to CSV.")
            return

        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['frame', 'player_id', 'team', 'jersey_number', 'distance_moved_since_last_frame', 'total_distance', 'x_pos', 'y_pos']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in self.csv_data:
                writer.writerow(row)

        print(f"CSV export completed: {output_path}")

    def calculate_real_world_distance(self, point1, point2, units='feet'):
        """
        Calculate real-world distance between two points in rink space
        point1, point2: coordinates in rink space (feet from center)
        units: 'feet' or 'meters'
        """
        dx = point1[0] - point2[0]
        dy = point1[1] - point2[1]

        distance = math.sqrt(dx*dx + dy*dy)

        if units == 'meters':
            distance *= self.homography_calc.feet_to_meters

        return distance

    def update_player_distances(self, tracker, tracked_detections, frame, frame_idx):
        """
        Update total distance traveled by each player and collect CSV data
        """
        for i, (xyxy, tracker_id) in enumerate(zip(tracked_detections.xyxy, tracked_detections.tracker_id)):
            # For a general model, treat all tracked objects as potentially players
            # Only track person-like objects (class 0 in COCO model) or all tracked objects
            current_img_point = (
                int((xyxy[0] + xyxy[2]) / 2),
                int((xyxy[1] + xyxy[3]) / 2)
            )

            # Transform to rink space
            current_rink_point = self.homography_calc.transform_point_to_rink_space(current_img_point)

            if current_rink_point is not None:
                self.player_positions[tracker_id] = current_rink_point

                # Store in history for KDE
                self.position_history[tracker_id].append(current_rink_point)

                # Update distance if we have a previous position
                prev_pos_key = f'prev_pos_{tracker_id}'
                prev_pos = self.prev_positions.get(tracker_id)

                distance_traveled = 0
                if prev_pos is not None:
                    distance_traveled = self.calculate_real_world_distance(prev_pos, current_rink_point)
                    self.player_distances[tracker_id] += distance_traveled

                # Get team and jersey number information
                team_info = tracker.player_teams.get(tracker_id, ('unknown', 'unknown'))
                team_name = team_info[0] if isinstance(team_info, tuple) else 'unknown'
                jersey_number = tracker.jersey_numbers.get(tracker_id, tracker_id)

                # Record data for CSV export
                csv_row = {
                    'frame': frame_idx,
                    'player_id': tracker_id,
                    'team': team_name,
                    'jersey_number': jersey_number,
                    'distance_moved_since_last_frame': distance_traveled,
                    'total_distance': self.player_distances[tracker_id],
                    'x_pos': current_rink_point[0],
                    'y_pos': current_rink_point[1]
                }
                self.csv_data.append(csv_row)

                # Update previous position
                self.prev_positions[tracker_id] = current_rink_point

    def calculate_puck_possession(self, player_positions, puck_position, frame_idx):
        """
        Determine which player has possession of the puck
        player_positions: dict of {tracker_id: (x, y) in rink space}
        puck_position: (x, y) in rink space
        """
        if puck_position is None:
            return None

        closest_player_id = None
        min_distance = float('inf')

        # Find closest player to puck
        for tracker_id, player_pos in player_positions.items():
            distance = self.calculate_real_world_distance(player_pos, puck_position)

            if distance < min_distance:
                min_distance = distance
                closest_player_id = tracker_id

        # Check if this constitutes possession
        if min_distance <= self.possession_threshold:
            # Update possession history
            if closest_player_id not in self.puck_control_history:
                self.puck_control_history[closest_player_id] = {'frames': [], 'start_frame': frame_idx}

            self.puck_control_history[closest_player_id]['frames'].append(frame_idx)

            # Check if this is sustained possession
            frames = self.puck_control_history[closest_player_id]['frames']
            recent_frames = [f for f in frames if frame_idx - f < self.possession_min_frames]

            if len(recent_frames) >= self.possession_min_frames:
                return closest_player_id

        return None

    def generate_kde_heatmap(self, tracker_id, sigma=15):
        """
        Generate a KDE heatmap for a specific player's movement
        tracker_id: ID of the player
        sigma: Standard deviation for Gaussian kernel
        """
        if tracker_id not in self.position_history or len(self.position_history[tracker_id]) < 10:
            # Not enough data for heatmap
            return np.zeros(self.heatmap_resolution, dtype=np.uint8)

        positions = list(self.position_history[tracker_id])

        # Create a grid for the heatmap
        heatmap = np.zeros(self.heatmap_resolution, dtype=np.float32)

        # Convert rink coordinates to pixel coordinates
        for rink_x, rink_y in positions:
            # Convert rink space to image space
            img_point = self.homography_calc.transform_point_from_rink_space((rink_x, rink_y))

            if img_point is not None:
                x, y = img_point

                # Check if the point is within the heatmap bounds
                if 0 <= x < self.heatmap_resolution[1] and 0 <= y < self.heatmap_resolution[0]:
                    # Add weight to the heatmap at this position
                    heatmap[y, x] += 1

        # Apply Gaussian smoothing
        heatmap = gaussian_filter(heatmap, sigma=sigma)

        # Normalize to 0-255 range
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max()) * 255

        return heatmap.astype(np.uint8)

    def get_player_statistics(self, tracker_id):
        """
        Get statistics for a specific player
        """
        stats = {
            'total_distance_feet': self.player_distances[tracker_id],
            'total_distance_meters': self.player_distances[tracker_id] * self.homography_calc.feet_to_meters,
            'avg_speed_fps': 0,  # Average speed in feet per second (would need time info)
            'position_history_count': len(self.position_history[tracker_id])
        }

        return stats

    def draw_rink_visualization(self, frame, tracker, tracked_detections, puck_position, possessed_player_id=None, selected_player_for_heatmap=None):
        """
        Draw the 2D rink visualization with player positions and puck
        """
        # Create a blank canvas for the rink visualization
        rink_canvas = np.ones((400, 600, 3), dtype=np.uint8) * 50  # Dark gray background

        # Draw rink outline
        cv2.rectangle(rink_canvas, (50, 50), (550, 350), (255, 255, 255), 2)

        # Draw center line
        cv2.line(rink_canvas, (300, 50), (300, 350), (255, 255, 255), 2)

        # Draw faceoff circles (approximate positions)
        cv2.circle(rink_canvas, (150, 150), 20, (255, 255, 255), 2)
        cv2.circle(rink_canvas, (450, 150), 20, (255, 255, 255), 2)
        cv2.circle(rink_canvas, (150, 250), 20, (255, 255, 255), 2)
        cv2.circle(rink_canvas, (450, 250), 20, (255, 255, 255), 2)

        # Draw goal areas
        cv2.rectangle(rink_canvas, (50, 150), (100, 250), (255, 255, 255), 2)  # Left goal
        cv2.rectangle(rink_canvas, (500, 150), (550, 250), (255, 255, 255), 2)  # Right goal

        # Convert and draw player positions with team coloring
        player_positions_rink = {}
        for i, (xyxy, tracker_id) in enumerate(zip(tracked_detections.xyxy, tracked_detections.tracker_id)):
            img_point = (int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2))
            rink_point = self.homography_calc.transform_point_to_rink_space(img_point)

            if rink_point is not None:
                player_positions_rink[tracker_id] = rink_point

                # Convert rink point to rink canvas coordinates
                # Map from rink feet (-89 to 89 length, -42.5 to 42.5 width) to canvas pixels
                x_canvas = int(((rink_point[0] + 89) / 178) * 500) + 50  # Scale to 500px width
                y_canvas = int(((rink_point[1] + 42.5) / 85) * 300) + 50  # Scale to 300px height

                # Get team information from tracker
                team_info = tracker.player_teams.get(tracker_id, ('unknown', 'unknown'))
                team_name = team_info[0] if isinstance(team_info, tuple) else 'unknown'

                # Draw player as a colored circle based on team
                if team_name == 'team_1':
                    color = (0, 0, 255)  # Red for team 1
                elif team_name == 'team_2':
                    color = (255, 0, 0)  # Blue for team 2
                elif team_name == 'goalie':
                    color = (0, 255, 0)  # Green for goalie
                elif team_name == 'referee':
                    color = (0, 255, 255)  # Yellow for referee
                else:
                    color = (0, 255, 0)  # Default green for unknown

                if tracker_id == possessed_player_id:
                    color = (0, 255, 255)  # Yellow if has puck possession

                cv2.circle(rink_canvas, (x_canvas, y_canvas), 8, color, -1)

                # Draw player ID number
                cv2.putText(rink_canvas, str(tracker_id), (x_canvas-5, y_canvas+5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw puck position
        if puck_position:
            puck_rink_point = self.homography_calc.transform_point_to_rink_space(puck_position)
            if puck_rink_point is not None:
                x_canvas = int(((puck_rink_point[0] + 89) / 178) * 500) + 50
                y_canvas = int(((puck_rink_point[1] + 42.5) / 85) * 300) + 50

                cv2.circle(rink_canvas, (x_canvas, y_canvas), 6, (0, 255, 255), -1)  # Yellow for puck

        # Add heatmap for selected player if specified
        if selected_player_for_heatmap is not None:
            heatmap = self.generate_kde_heatmap(selected_player_for_heatmap)
            if heatmap.size > 0:
                # Overlay heatmap on rink canvas
                heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                # Resize heatmap to fit rink canvas
                heatmap_resized = cv2.resize(heatmap_color, (500, 300))

                # Blend with rink canvas
                alpha = 0.3
                rink_roi = rink_canvas[50:350, 50:550]
                blended = cv2.addWeighted(rink_roi, 1-alpha, heatmap_resized, alpha, 0)
                rink_canvas[50:350, 50:550] = blended

        # Add title
        cv2.putText(rink_canvas, 'Hockey Rink View', (200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Add team legend
        cv2.rectangle(rink_canvas, (10, 360), (250, 400), (0, 0, 0), -1)  # Background
        cv2.putText(rink_canvas, 'Teams:', (20, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(rink_canvas, 'Dark Team: Red', (70, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(rink_canvas, 'Light Team: Blue', (150, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        return rink_canvas

    def process_frame_analytics(self, frame, tracker, tracked_detections, puck_position, frame_idx, selected_player_for_heatmap=None):
        """
        Process a single frame to update analytics
        """
        # Update player distances
        self.update_player_distances(tracker, tracked_detections, frame, frame_idx)

        # Calculate puck possession
        possessed_player_id = self.calculate_puck_possession(self.player_positions, puck_position, frame_idx)

        # Draw rink visualization
        rink_vis = self.draw_rink_visualization(
            frame, tracker, tracked_detections, puck_position,
            possessed_player_id, selected_player_for_heatmap
        )

        return rink_vis, possessed_player_id