import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

class HockeyVisualization:
    def __init__(self, rink_width=800, rink_height=400):
        """
        Initialize visualization module for hockey analytics
        """
        self.rink_width = rink_width
        self.rink_height = rink_height

        # Pre-compute standard rink layout
        self.rink_layout = self.create_rink_layout()

    def create_rink_layout(self):
        """
        Create a standard hockey rink layout
        """
        # Create a canvas for the rink
        rink = np.zeros((self.rink_height, self.rink_width, 3), dtype=np.uint8)

        # Rink dimensions (scaled to fit canvas)
        margin_x = 50
        margin_y = 50
        rink_field_width = self.rink_width - 2 * margin_x
        rink_field_height = self.rink_height - 2 * margin_y

        # Draw rink boundary
        cv2.rectangle(rink, (margin_x, margin_y),
                     (self.rink_width - margin_x, self.rink_height - margin_y),
                     (255, 255, 255), 2)

        # Draw center line
        center_x = self.rink_width // 2
        cv2.line(rink, (center_x, margin_y), (center_x, self.rink_height - margin_y), (255, 255, 255), 2)

        # Draw center circle
        cv2.circle(rink, (center_x, self.rink_height // 2), 30, (255, 255, 255), 2)

        # Draw faceoff circles (4 total: center and 2 in each attacking zone)
        faceoff_positions = [
            (center_x, self.rink_height // 2),  # Center
            (center_x - 200, self.rink_height // 2 - 80),  # Left upper
            (center_x - 200, self.rink_height // 2 + 80),  # Left lower
            (center_x + 200, self.rink_height // 2 - 80),  # Right upper
            (center_x + 200, self.rink_height // 2 + 80),  # Right lower
        ]

        for pos in faceoff_positions:
            cv2.circle(rink, pos, 15, (255, 255, 255), 2)
            # Draw hash marks around circle
            cv2.rectangle(rink, (pos[0]-20, pos[1]-5), (pos[0]+20, pos[1]+5), (255, 255, 255), -1)

        # Draw goal areas
        # Left goal
        cv2.rectangle(rink, (margin_x, self.rink_height//2 - 60),
                     (margin_x + 60, self.rink_height//2 + 60), (255, 255, 255), 2)
        # Right goal
        cv2.rectangle(rink, (self.rink_width - margin_x - 60, self.rink_height//2 - 60),
                     (self.rink_width - margin_x, self.rink_height//2 + 60), (255, 255, 255), 2)

        # Draw blue lines (20 feet from center in each direction)
        blue_line_offset = int((20 / 100) * rink_field_width)  # Scale based on field width
        cv2.line(rink, (center_x - blue_line_offset, margin_y),
                (center_x - blue_line_offset, self.rink_height - margin_y), (0, 0, 255), 2)
        cv2.line(rink, (center_x + blue_line_offset, margin_y),
                (center_x + blue_line_offset, self.rink_height - margin_y), (0, 0, 255), 2)

        return rink

    def plot_on_rink(self, positions_dict, puck_pos=None, possessed_player_id=None,
                     selected_player_heatmap=None, heatmap_data=None):
        """
        Plot player and puck positions on the rink layout

        Args:
            positions_dict: dict of {player_id: (x, y) in rink coords}
            puck_pos: (x, y) puck position in rink coords
            possessed_player_id: ID of player with puck possession
            selected_player_heatmap: ID of player to show heatmap for
            heatmap_data: precomputed heatmap data for the selected player
        """
        # Start with the base rink layout
        vis_frame = self.rink_layout.copy()

        # Convert rink coordinates to pixel coordinates
        def rink_to_pixel(pos):
            if pos is None:
                return None
            # Assuming rink coords are from -89 to 89 feet (length) and -42.5 to 42.5 feet (width)
            # Map to our canvas dimensions
            rink_min_x, rink_max_x = -89, 89
            rink_min_y, rink_max_y = -42.5, 42.5

            x_norm = (pos[0] - rink_min_x) / (rink_max_x - rink_min_x)
            y_norm = (pos[1] - rink_min_y) / (rink_max_y - rink_min_y)

            pixel_x = int(50 + x_norm * (self.rink_width - 100))
            pixel_y = int(50 + (1 - y_norm) * (self.rink_height - 100))  # Flip Y axis

            return (pixel_x, pixel_y)

        # Add heatmap if available
        if heatmap_data is not None and selected_player_heatmap is not None:
            # Apply heatmap as overlay
            heatmap_display = cv2.applyColorMap((heatmap_data * 255).astype(np.uint8), cv2.COLORMAP_JET)
            alpha = 0.3
            vis_frame = cv2.addWeighted(vis_frame, 1 - alpha, heatmap_display, alpha, 0)

        # Plot players
        for player_id, rink_pos in positions_dict.items():
            pixel_pos = rink_to_pixel(rink_pos)
            if pixel_pos:
                # Different colors for different roles
                color = (0, 255, 0)  # Default green for regular players
                if player_id == possessed_player_id:
                    color = (0, 255, 255)  # Yellow for player with puck
                    thickness = 3
                else:
                    thickness = 2

                cv2.circle(vis_frame, pixel_pos, 10, color, thickness)

                # Add player ID
                cv2.putText(vis_frame, str(player_id),
                           (pixel_pos[0]-10, pixel_pos[1]-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Plot puck
        if puck_pos:
            pixel_puck_pos = rink_to_pixel(puck_pos)
            if pixel_puck_pos:
                cv2.circle(vis_frame, pixel_puck_pos, 6, (0, 255, 255), -1)  # Yellow filled circle for puck
                cv2.putText(vis_frame, 'P', pixel_puck_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add legend
        cv2.putText(vis_frame, 'Players', (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis_frame, 'With Puck', (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return vis_frame

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

class HockeyVisualization:
    def __init__(self):
        """
        Initialize visualization module for hockey analytics (without rink view)
        """
        pass

    def create_single_view(self, original_frame, tracked_detections, tracker):
        """
        Create a single view with original video and player information
        """
        annotated_frame = original_frame.copy()

        # Draw bounding boxes with team colors
        for i, (xyxy, tracker_id) in enumerate(zip(tracked_detections.xyxy, tracked_detections.tracker_id)):
            x1, y1, x2, y2 = map(int, xyxy)

            # Get team information
            team_info = tracker.player_teams.get(tracker_id, ('unknown', 'unknown'))
            team_name = team_info[0] if isinstance(team_info, tuple) else 'unknown'

            # Choose color based on team
            if team_name == 'team_1':
                color = (255, 0, 0)  # Blue for team 1
            elif team_name == 'team_2':
                color = (0, 0, 255)  # Red for team 2
            elif team_name == 'goalie':
                color = (0, 255, 0)  # Green for goalie
            elif team_name == 'referee':
                color = (0, 255, 255)  # Yellow for referee
            else:
                color = (255, 255, 255)  # White for unknown

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Get jersey number
            jersey_num = tracker.jersey_numbers.get(tracker_id, str(tracker_id))

            # Determine player status based on role
            if team_name == 'referee':
                status = "Ref"
            elif team_name == 'goalie':
                status = "Goalie"
            else:
                status = "Player"

            # Draw label with team color, jersey number, and status
            label = f"{jersey_num} ({status})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + 30

            cv2.putText(annotated_frame, label,
                      (x1, label_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Add team color legend to the frame
        self.add_team_legend(annotated_frame, tracker)

        return annotated_frame

    def add_team_legend(self, frame, tracker):
        """
        Add team color legend to the frame
        """
        # Find the established teams from tracker
        team_colors = {}
        for team_info in tracker.player_teams.values():
            if isinstance(team_info, tuple) and len(team_info) >= 2:
                team_name = team_info[0]
                if team_name.startswith('team_'):
                    team_colors[team_name] = team_info[1]  # color info

        y_offset = 20
        for i, (team_name, _) in enumerate(sorted(team_colors.items())):
            if team_name == 'team_1':
                color = (255, 0, 0)  # Blue
                team_label = "Team 1"
            elif team_name == 'team_2':
                color = (0, 0, 255)  # Red
                team_label = "Team 2"
            else:
                color = (128, 128, 128)  # Gray for others
                team_label = team_name.replace('_', ' ').title()

            # Draw color rectangle
            cv2.rectangle(frame, (10, y_offset), (30, y_offset + 20), color, -1)

            # Draw team label
            cv2.putText(frame, f"{team_label}",
                      (40, y_offset + 15),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            y_offset += 30

    def create_side_by_side_view(self, original_frame, rink_frame, scale_factor=0.7):
        """
        Create a side-by-side view of original video and rink visualization
        This method is kept for compatibility but will return just the original frame
        """
        # For now, just return the original frame since we're not showing rink view
        return original_frame