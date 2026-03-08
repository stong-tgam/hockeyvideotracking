import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict, deque
import math

class TeamClassifier:
    """
    Classifies players into teams based on dominant color analysis
    """
    def __init__(self):
        self.team_colors = {}
        self.team_avg_colors = {}  # Store average color for each team
        self.processing_initial_analysis = True  # Flag to indicate initial team analysis phase
        self.frame_team_analysis_counter = 0
        self.max_analysis_frames = 30  # Analyze first 30 frames to establish team colors
        self.bottom_half_only = True
        self.processing_team_assignment = True

    def calculate_average_color(self, frame, bbox):
        """
        Calculate average RGB color in the jersey area of a player
        Focus on the bottom half of the frame (where players typically are)
        """
        x1, y1, x2, y2 = map(int, bbox)

        # Ensure coordinates are within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return None

        # Crop the player region
        player_region = frame[y1:y2, x1:x2]

        # Focus on the middle horizontal stripe of the player (chest/jersey area)
        h_start = int(player_region.shape[0] * 0.3)
        h_end = int(player_region.shape[0] * 0.7)

        if h_start < h_end:
            jersey_area = player_region[h_start:h_end, :]
            avg_color = np.mean(jersey_area, axis=(0, 1))
            return tuple(avg_color.astype(int))

        # If the player is too small, just return the average of the whole region
        avg_color = np.mean(player_region, axis=(0, 1))
        return tuple(avg_color.astype(int))

    def rgb_to_hsv(self, rgb):
        """
        Convert RGB color to HSV for better color comparison
        """
        # Convert single RGB value to HSV
        rgb_normalized = np.array([[[rgb[0], rgb[1], rgb[2]]]], dtype=np.uint8)
        hsv = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2HSV)
        return tuple(hsv[0, 0])

    def color_distance(self, color1, color2):
        """
        Calculate distance between two colors in HSV space
        """
        h1, s1, v1 = self.rgb_to_hsv(color1)
        h2, s2, v2 = self.rgb_to_hsv(color2)

        # Weight hue more heavily as it's the most important for color distinction
        return abs(h1 - h2) * 0.6 + abs(s1 - s2) * 0.3 + abs(v1 - v2) * 0.1

    def classify_player_by_color(self, frame, bbox, tracker_id):
        """
        Classify a player based on their jersey color
        """
        avg_color = self.calculate_average_color(frame, bbox)

        if avg_color is None:
            # Always return a tuple with 3 values to prevent the unpacking error
            return "unknown", "unknown", (128, 128, 128)  # Return a default gray color

        # During initial analysis phase, collect team color information across all players
        if self.processing_initial_analysis:
            # Check if we already have 2 team colors established
            if len(self.team_avg_colors) < 2:
                # Look for distinctly different colors to establish the two teams
                is_different_from_existing = True
                for team_name, team_color in self.team_avg_colors.items():
                    if self.color_distance(avg_color, team_color) < 50:  # Similar color threshold
                        is_different_from_existing = False
                        # Add this as evidence for this team
                        return team_name, "unknown", avg_color

                # If we found a sufficiently different color and room for a new team
                if is_different_from_existing and len(self.team_avg_colors) < 2:
                    new_team_name = f"team_{len(self.team_avg_colors) + 1}"
                    self.team_avg_colors[new_team_name] = avg_color
                    return new_team_name, "unknown", avg_color
            else:
                # We've identified 2 teams, now move to assignment phase
                self.processing_initial_analysis = False

        # Regular assignment phase - assign to closest known team
        if len(self.team_avg_colors) == 0:
            # If no team colors established yet, assign to first team temporarily
            team_name = "team_1"
            self.team_avg_colors[team_name] = avg_color
            return team_name, "unknown", avg_color

        closest_team = "unknown"
        min_distance = float('inf')

        for team_name, team_color in self.team_avg_colors.items():
            dist = self.color_distance(avg_color, team_color)
            if dist < min_distance:
                min_distance = dist
                closest_team = team_name

        return closest_team, "unknown", avg_color

    def get_dominant_color(self, frame, bbox):
        """
        Extract dominant color from a player's jersey region
        """
        avg_color = self.calculate_average_color(frame, bbox)
        if avg_color is None:
            return 'unknown'
        return avg_color

    def classify_team(self, frame, bbox, tracker_id, current_teams, class_id=None):
        """
        Classify player into team based on jersey color
        class_id: YOLO class ID for this detection (0=center ice, 1=faceoff, 2=goal, 3=goalie, 4=player, 5=puck, 6=referee)
        """
        # If it's definitely a referee (class_id 6), return referee classification
        if class_id == 6:  # Referee class
            return 'referee', 'special_referee'

        # If it's definitely a goaltender (class_id 3), return goalie classification
        if class_id == 3:  # Goaltender class
            return 'goalie', 'goalie_equipment'

        # For players, analyze jersey color
        if class_id == 4:  # Player class
            team_name, role, avg_color = self.classify_player_by_color(frame, bbox, tracker_id)
            return team_name, avg_color

        # For other detections, return unknown
        return 'unknown', 'unknown'

    def cluster_by_similarity(self, new_color, current_teams):
        """
        Cluster new player with existing team based on other players' classifications
        """
        return 'team_unknown'

class HockeyTracker:
    def __init__(self, model_path):
        """
        Initialize the hockey tracker with YOLO model and tracking algorithms
        """
        self.model = YOLO(model_path)

        # Check if the model has the expected hockey classes
        self.is_hockey_model = self._check_if_hockey_model()

        # Use ByteTrack for tracking
        self.tracker = sv.ByteTrack()

        # Store jersey numbers for players (to be detected automatically)
        self.jersey_numbers = {}

        # Store team information for players
        self.player_teams = {}
        self.team_classifier = TeamClassifier()

        # Store trajectories for visualization
        self.player_trajectories = defaultdict(lambda: deque(maxlen=30))
        self.puck_trajectory = deque(maxlen=15)

        # For puck interpolation
        self.puck_positions_history = deque(maxlen=10)
        self.last_known_puck_position = None

        # Track initialization state - this is no longer a single-pass initialization
        # as team assignment happens continuously for new tracker IDs
        self.initialization_complete = True  # Set to True initially since assignment happens every frame
        self.initialization_frame = None

        print(f"Hockey model detection: {self.is_hockey_model}")

    def _check_if_hockey_model(self):
        """
        Check if the loaded model has hockey-specific classes
        """
        try:
            # Get model information to see class names
            # Check if it has hockey-specific classes
            names_str = ' '.join(self.model.names.values()).lower()
            hockey_keywords = ['player', 'puck', 'goal', 'goaltender', 'referee', 'center', 'faceoff', 'net']

            return any(keyword in names_str for keyword in hockey_keywords)
        except:
            # If there's an issue checking, assume it's not a hockey model
            return False

    def detect(self, frame):
        """
        Run YOLO detection on the frame
        Returns detections with class information
        """
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        return detections

    def detect_jersey_numbers(self, frame, player_detections):
        """
        Detect jersey numbers from player bounding boxes
        """
        # This would be replaced with actual OCR in a production system
        # For now, we'll assign jersey numbers based on tracker IDs
        for tracker_id in player_detections.tracker_id:
            if tracker_id not in self.jersey_numbers:
                # For hockey players, assign number based on tracker ID
                if len(self.jersey_numbers) < 20:  # Reasonable number for hockey team
                    self.jersey_numbers[tracker_id] = str((tracker_id % 99) + 1)  # Numbers 1-99
                else:
                    self.jersey_numbers[tracker_id] = str(tracker_id)

    def assign_jersey_numbers(self, frame, tracked_detections):
        """
        Assign jersey numbers based on model classes or detection patterns
        Also classify teams based on jersey colors for new tracker IDs
        """
        if self.is_hockey_model:
            # Filter for hockey-specific classes that have jersey numbers
            # Typically players (not goaltenders/referees) wear numbered jerseys
            # Loop through tracked_detections to process each detection
            for i, (xyxy, tracker_id) in enumerate(zip(tracked_detections.xyxy, tracked_detections.tracker_id)):
                # Check if this tracker_id is already assigned - only assign new ones
                if tracker_id not in self.jersey_numbers or tracker_id not in self.player_teams:
                    # Get class_id for this specific detection
                    class_id = tracked_detections.class_id[i] if i < len(tracked_detections.class_id) else None

                    # Classify based on class_id
                    if class_id == 4:  # Player class
                        # Check if we need to assign jersey number for this new player
                        if tracker_id not in self.jersey_numbers:
                            # Assign jersey number based on existing numbers to avoid duplicates
                            assigned_number = len([k for k, v in self.jersey_numbers.items() if not v.startswith(('G', 'R'))]) + 1
                            self.jersey_numbers[tracker_id] = str(assigned_number)

                        # Classify team based on jersey color only if not already classified
                        if tracker_id not in self.player_teams:
                            team_name, jersey_color = self.team_classifier.classify_team(
                                frame, xyxy, tracker_id, self.player_teams, class_id=4  # 4 is player class
                            )
                            self.player_teams[tracker_id] = (team_name, jersey_color)

                    elif class_id == 3:  # Goaltender class
                        # Check if we need to assign identifier for this new goalie
                        if tracker_id not in self.jersey_numbers:
                            self.jersey_numbers[tracker_id] = "G"

                        # Classify team for goalie only if not already classified
                        if tracker_id not in self.player_teams:
                            team_name, jersey_color = self.team_classifier.classify_team(
                                frame, xyxy, tracker_id, self.player_teams, class_id=3  # 3 is goalie class
                            )
                            self.player_teams[tracker_id] = (team_name, jersey_color)

                    elif class_id == 6:  # Referee class
                        # Check if we need to assign identifier for this new referee
                        if tracker_id not in self.jersey_numbers:
                            self.jersey_numbers[tracker_id] = "REF"

                        # Classify team for referee only if not already classified
                        if tracker_id not in self.player_teams:
                            team_name, jersey_color = self.team_classifier.classify_team(
                                frame, xyxy, tracker_id, self.player_teams, class_id=6  # 6 is referee class
                            )
                            self.player_teams[tracker_id] = (team_name, jersey_color)

        else:
            # For general model, process each tracked detection
            for i, (xyxy, tracker_id) in enumerate(zip(tracked_detections.xyxy, tracked_detections.tracker_id)):
                # Check if this tracker_id is already assigned - only assign new ones
                if tracker_id not in self.jersey_numbers or tracker_id not in self.player_teams:
                    # Check if we need to assign jersey number for this new person
                    if tracker_id not in self.jersey_numbers:
                        self.jersey_numbers[tracker_id] = str(len(self.jersey_numbers) + 1)

                    # Classify team for general person detection only if not already classified
                    if tracker_id not in self.player_teams:
                        team_name, jersey_color = self.team_classifier.classify_team(
                            frame, xyxy, tracker_id, self.player_teams
                        )
                        self.player_teams[tracker_id] = (team_name, jersey_color)

        # Print occasional updates but not every frame
        if hasattr(self, '_last_print_frame'):
            if hasattr(self, 'frame_count'):
                self.frame_count += 1
                if self.frame_count % 30 == 0:  # Print every 30 frames
                    print(f"Frame {self.frame_count}: Jersey numbers assigned: {dict(list(self.jersey_numbers.items())[:10])}")
                    print(f"Team assignments: {dict(list(self.player_teams.items())[:10])}")
        else:
            self.frame_count = 0
            print(f"Jersey numbers assigned: {dict(list(self.jersey_numbers.items())[:10])}")  # Show first 10
            print(f"Team assignments: {dict(list(self.player_teams.items())[:10])}")  # Show first 10
            self.initialization_complete = True  # Set initialization complete when first assignment happens

    def interpolate_puck_position(self, frame, detections, current_frame_idx):
        """
        Interpolate puck position when it's not detected
        Uses history to predict location during occlusions
        """
        if self.is_hockey_model:
            puck_detections = detections[detections.class_id == 5]  # Puck class in hockey model
        else:
            # For general model, we might not have a puck class, so use alternative
            puck_detections = detections[:0]  # Empty detection as fallback

        if len(puck_detections.xyxy) > 0:
            # Puck detected, store the position
            puck_center = (
                int((puck_detections.xyxy[0][0] + puck_detections.xyxy[0][2]) / 2),
                int((puck_detections.xyxy[0][1] + puck_detections.xyxy[0][3]) / 2)
            )
            self.puck_positions_history.append(puck_center)
            self.last_known_puck_position = puck_center
            return puck_center
        elif self.last_known_puck_position is not None and len(self.puck_positions_history) >= 2:
            # Puck not detected, interpolate based on history
            # Use linear extrapolation from last known positions
            if len(self.puck_positions_history) >= 2:
                pos1 = self.puck_positions_history[-1]
                pos2 = self.puck_positions_history[-2]

                # Calculate direction vector
                dx = pos1[0] - pos2[0]
                dy = pos1[1] - pos2[1]

                # Extrapolate next position (assuming constant velocity)
                predicted_pos = (pos1[0] + dx, pos1[1] + dy)

                # Apply constraints (keep within frame)
                h, w = frame.shape[:2]
                predicted_pos = (
                    max(0, min(w, predicted_pos[0])),
                    max(0, min(h, predicted_pos[1]))
                )

                return predicted_pos
            else:
                return self.last_known_puck_position
        else:
            return self.last_known_puck_position

    def track(self, frame, frame_idx=0):
        """
        Process a frame with detection and tracking
        """
        # Run detection
        detections = self.detect(frame)

        # Apply tracking
        tracked_detections = self.tracker.update_with_detections(detections)

        # Handle jersey number assignment and team classification on every frame for new trackers
        self.assign_jersey_numbers(frame, tracked_detections)

        # Get interpolated puck position
        puck_position = self.interpolate_puck_position(frame, detections, frame_idx)

        # Store trajectories
        for i, (xyxy, tracker_id) in enumerate(zip(tracked_detections.xyxy, tracked_detections.tracker_id)):
            center_point = (
                int((xyxy[0] + xyxy[2]) / 2),
                int((xyxy[1] + xyxy[3]) / 2)
            )

            # Store trajectory for tracked objects (players, goaltenders, refs)
            if self.is_hockey_model:
                if i < len(tracked_detections.class_id):
                    class_id = tracked_detections.class_id[i]
                    # Only track players, goaltenders, and refs (classes 3, 4, 6)
                    if class_id in [3, 4, 6]:
                        self.player_trajectories[tracker_id].append(center_point)
            else:
                # For general model, track all detections
                self.player_trajectories[tracker_id].append(center_point)

        if puck_position:
            self.puck_trajectory.append(puck_position)

        return tracked_detections, puck_position

    def visualize(self, frame, tracked_detections, puck_position=None):
        """
        Visualize tracking results on frame
        """
        # Annotators
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # Prepare labels with jersey numbers or tracker IDs and class information
        labels = []
        for i, tracker_id in enumerate(tracked_detections.tracker_id):
            jersey_num = self.jersey_numbers.get(tracker_id, str(tracker_id))

            # Add class name if using hockey model
            if self.is_hockey_model and i < len(tracked_detections.class_id):
                class_id = tracked_detections.class_id[i]
                class_names = {0: "CI", 1: "FD", 2: "GF", 3: "G", 4: "P", 5: "PK", 6: "R"}  # Center Ice, Faceoff Dots, Goal Frame, Goaltender, Player, Puck, Referee
                class_label = class_names.get(class_id, "")
                labels.append(f"{class_label}#{jersey_num}")
            else:
                labels.append(f"#{jersey_num}")

        # Annotate frame
        annotated_frame = frame.copy()

        # Color code by team
        for i, (xyxy, tracker_id) in enumerate(zip(tracked_detections.xyxy, tracked_detections.tracker_id)):
            # Get team information
            team_info = self.player_teams.get(tracker_id, ('unknown', 'unknown'))
            team_name = team_info[0] if isinstance(team_info, tuple) else 'unknown'

            # Choose color based on team
            if team_name == 'team_1':
                color = sv.Color(0, 0, 255)  # Red for team 1 (often darker colored)
            elif team_name == 'team_2':
                color = sv.Color(255, 0, 0)  # Blue for team 2 (often lighter colored)
            elif team_name == 'goalie':
                color = sv.Color(0, 255, 0)  # Green for goalie
            elif team_name == 'referee':
                color = sv.Color(0, 255, 255)  # Yellow for referee
            else:
                color = sv.Color(255, 255, 255)  # White for unknown

            # Draw bounding box with team color
            annotated_frame = cv2.rectangle(
                annotated_frame,
                (int(xyxy[0]), int(xyxy[1])),
                (int(xyxy[2]), int(xyxy[3])),
                color.as_bgr(), 2
            )

        # Add labels
        for i, (xyxy, tracker_id) in enumerate(zip(tracked_detections.xyxy, tracked_detections.tracker_id)):
            # Get team information
            team_info = self.player_teams.get(tracker_id, ('unknown', 'unknown'))
            team_name = team_info[0] if isinstance(team_info, tuple) else 'unknown'

            # Choose text color based on team
            if team_name == 'team_1':
                color = sv.Color(0, 0, 255)  # Red for team 1 (often darker colored)
            elif team_name == 'team_2':
                color = sv.Color(255, 0, 0)  # Blue for team 2 (often lighter colored)
            elif team_name == 'goalie':
                color = sv.Color(0, 255, 0)  # Green for goalie
            elif team_name == 'referee':
                color = sv.Color(0, 255, 255)  # Yellow for referee
            else:
                color = sv.Color(255, 255, 255)  # White for unknown

            # Draw label
            label_text = labels[i] if i < len(labels) else str(tracker_id)
            cv2.putText(
                annotated_frame,
                label_text,
                (int(xyxy[0]), int(xyxy[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color.as_bgr(),
                2
            )

        # Add puck visualization if available
        if puck_position:
            cv2.circle(annotated_frame, puck_position, 10, (0, 255, 255), -1)  # Yellow circle for puck
            cv2.putText(annotated_frame, 'Puck',
                       (puck_position[0]+15, puck_position[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Add player trajectories
        for tracker_id, trajectory in self.player_trajectories.items():
            if len(trajectory) > 1:
                # Get team color for trajectory
                team_info = self.player_teams.get(tracker_id, ('unknown', 'unknown'))
                team_name = team_info[0] if isinstance(team_info, tuple) else 'unknown'

                if team_name == 'team_1':
                    traj_color = (0, 0, 255)  # Red for team 1 (often darker colored)
                elif team_name == 'team_2':
                    traj_color = (255, 0, 0)  # Blue for team 2 (often lighter colored)
                elif team_name == 'goalie':
                    traj_color = (0, 255, 0)  # Green for goalie
                elif team_name == 'referee':
                    traj_color = (0, 255, 255)  # Yellow for referee
                else:
                    traj_color = (0, 255, 0)  # Default green

                points = np.array(trajectory)
                for i in range(1, len(points)):
                    cv2.line(annotated_frame, tuple(points[i-1]), tuple(points[i]), traj_color, 2)

        # Add puck trajectory
        if len(self.puck_trajectory) > 1:
            points = list(self.puck_trajectory)
            for i in range(1, len(points)):
                cv2.line(annotated_frame, points[i-1], points[i], (0, 255, 255), 2)  # Yellow for puck

        return annotated_frame