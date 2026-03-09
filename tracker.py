import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict, deque
import math
from sklearn.cluster import KMeans

class TeamClassifier:
    """
    Classifies players into teams based on dominant color analysis using K-means clustering
    """
    def __init__(self):
        self.team_colors = {}
        self.team_avg_colors = {}  # Store average color for each team
        self.processing_initial_analysis = True  # Flag to indicate initial team analysis phase
        self.frame_team_analysis_counter = 0
        self.max_analysis_frames = 30  # Analyze first 30 frames to establish team colors
        self.bottom_half_only = True
        self.processing_team_assignment = True

        # K-means clustering variables
        self.team_1_color = None  # Baseline team 1 color (from k-means clustering)
        self.team_2_color = None  # Baseline team 2 color (from k-means clustering)
        self.kmeans_initialized = False  # Whether baseline team colors have been established

    def extract_torso_pixels(self, frame, bbox):
        """
        Step 1: Extract torso pixels from player bounding box
        Crop the image to isolate only the upper middle half (the torso)
        This is to avoid the ice, legs, skates, and sticks.
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

        # Focus on the upper middle half (torso region) - this is the "upper middle half" (chest/jersey area)
        h_start = int(player_region.shape[0] * 0.25)  # Start at 25% height
        h_end = int(player_region.shape[0] * 0.75)    # End at 75% height

        if h_start < h_end:
            torso_region = player_region[h_start:h_end, :]
            # Reshape to list of pixels for k-means clustering
            pixels = torso_region.reshape(-1, 3)  # Reshape to (N, 3) where N is number of pixels
            return pixels.astype(np.float32)

        return None

    def initialize_team_baseline_colors(self, frames_and_detections):
        """
        Step 2: Find Team Baselines using K-Means Clustering
        Take all extracted torso pixels and use K-Means clustering with k=2 to find the two dominant cluster centers.
        Store these two cluster centers as team_1_color and team_2_color.
        """
        from sklearn.cluster import KMeans

        all_torso_pixels = []

        # Collect torso pixels from all frames and detections
        for frame, detections in frames_and_detections:
            for xyxy in detections.xyxy:
                pixels = self.extract_torso_pixels(frame, xyxy)
                if pixels is not None:
                    all_torso_pixels.append(pixels)

        if len(all_torso_pixels) == 0:
            return False

        # Concatenate all pixels from all detections
        all_pixels = np.vstack(all_torso_pixels)

        # Apply color filtering to distinguish jersey colors from background
        all_pixels_uint8 = all_pixels.astype(np.uint8)

        # Filter to keep likely jersey colors
        not_too_bright = ~np.all(all_pixels_uint8 >= [245, 245, 245], axis=1)  # Filter out very bright (likely ice reflections)
        not_too_dark = ~np.all(all_pixels_uint8 <= [30, 30, 30], axis=1)       # Filter out very dark (likely helmets/shadows)

        # Calculate brightness to filter out background
        brightness = np.mean(all_pixels_uint8, axis=1)
        reasonable_brightness = (brightness > 50) & (brightness < 230)

        # Combine filters
        valid_pixels_mask = not_too_dark & not_too_bright & reasonable_brightness
        filtered_pixels = all_pixels[valid_pixels_mask]

        if len(filtered_pixels) < 2:
            return False

        # Apply K-means clustering with k=2 to find team baseline colors
        n_clusters = min(2, len(filtered_pixels))  # Ensure we don't have more clusters than pixels
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(filtered_pixels)

        # Store the two cluster centers as team baseline colors
        cluster_centers = kmeans.cluster_centers_

        if len(cluster_centers) >= 1:
            self.team_1_color = cluster_centers[0].astype(int)
        if len(cluster_centers) >= 2:
            self.team_2_color = cluster_centers[1].astype(int)

        self.kmeans_initialized = True
        return True

    def assign_player_to_team(self, frame, bbox):
        """
        Step 3: Frame-by-Frame Assignment
        Crop bounding box to torso region, apply color masks, run k=1 clustering,
        convert to LAB color space, calculate color distances, return assigned team.
        """
        from sklearn.cluster import KMeans

        if not self.kmeans_initialized:
            # If baselines aren't set yet, return unknown
            return "unknown", "unknown", (128, 128, 128)

        # Step 1: Crop bounding box to torso region
        pixels = self.extract_torso_pixels(frame, bbox)
        if pixels is None:
            return "unknown", "unknown", (128, 128, 128)

        # Step 2: Apply color mask to filter out pure white (ice/edge cases) and pure black/dark gray (helmets, shadows, gloves)
        # Convert pixels to uint8 for comparison
        pixels_uint8 = pixels.astype(np.uint8)

        # Create mask for acceptable jersey colors
        # For white jerseys: values 180-255 in all channels should be kept
        # For black jerseys: values 0-70 in all channels should be kept
        # Filter out extreme values that are likely ice (very bright) or helmets (very dark)
        not_too_bright = ~np.all(pixels_uint8 >= [245, 245, 245], axis=1)  # Filter out very bright (likely ice reflections)
        not_too_dark = ~np.all(pixels_uint8 <= [30, 30, 30], axis=1)       # Filter out very dark (likely helmets/shadows)

        # For jersey colors, we want to keep colors that are neither too dark nor extremely bright
        # But still include white jerseys (which can be quite bright but not as bright as ice)
        jersey_mask = not_too_dark & not_too_bright

        # Additional filter to keep colors that look like jerseys
        # Calculate brightness to distinguish from background
        brightness = np.mean(pixels_uint8, axis=1)
        # Keep colors that have reasonable brightness for jerseys (not too dim, not ice-bright)
        reasonable_brightness = (brightness > 50) & (brightness < 230)

        # Combine masks
        valid_pixels_mask = jersey_mask & reasonable_brightness
        valid_pixels = pixels[valid_pixels_mask]

        if len(valid_pixels) == 0:
            return "unknown", "unknown", (128, 128, 128)

        # Step 3: Run K-Means with k=1 on remaining unmasked pixels to find the player's dominant jersey color
        if len(valid_pixels) < 1:
            return "unknown", "unknown", (128, 128, 128)

        kmeans = KMeans(n_clusters=1, random_state=42, n_init=10)
        kmeans.fit(valid_pixels)
        player_dominant_color = kmeans.cluster_centers_[0].astype(int)

        # Step 4: Convert both player's dominant color and team baseline colors from BGR/RGB to LAB color space
        player_lab = self.rgb_to_lab(player_dominant_color)
        team1_lab = self.rgb_to_lab(self.team_1_color if self.team_1_color is not None else [128, 128, 128])
        team2_lab = self.rgb_to_lab(self.team_2_color if self.team_2_color is not None else [128, 128, 128])

        # Step 5: Calculate the color distance between the player and the two baselines using LAB values
        dist_to_team1 = self.lab_color_distance(player_lab, team1_lab)
        dist_to_team2 = self.lab_color_distance(player_lab, team2_lab)

        # Step 6: Return the assigned team (the one with the shortest distance)
        if dist_to_team1 <= dist_to_team2:
            return "team_1", "jersey_color_1", tuple(player_dominant_color)
        else:
            return "team_2", "jersey_color_2", tuple(player_dominant_color)

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

    def rgb_to_lab(self, rgb):
        """
        Convert RGB color to LAB color space
        """
        # Convert RGB to BGR (OpenCV format)
        bgr = np.uint8([[list(reversed(rgb))]])  # Reverse RGB to BGR
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        return tuple(lab[0, 0])

    def lab_color_distance(self, lab1, lab2):
        """
        Calculate distance between two colors in LAB space using Euclidean distance
        """
        dL = lab1[0] - lab2[0]
        da = lab1[1] - lab2[1]
        db = lab1[2] - lab2[2]
        return math.sqrt(dL*dL + da*da + db*db)

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
        Classify a player based on their jersey color using K-means approach
        """
        # Use the new K-means based team assignment
        team_name, role, avg_color = self.assign_player_to_team(frame, bbox)

        if team_name == "unknown":
            # Fallback to old method if K-means fails
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
                    for team_name_key, team_color in self.team_avg_colors.items():
                        if self.color_distance(avg_color, team_color) < 50:  # Similar color threshold
                            is_different_from_existing = False
                            # Add this as evidence for this team
                            return team_name_key, "unknown", avg_color

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
                team_name_result = "team_1"
                self.team_avg_colors[team_name_result] = avg_color
                return team_name_result, "unknown", avg_color

            closest_team = "unknown"
            min_distance = float('inf')

            for team_name_key, team_color in self.team_avg_colors.items():
                dist = self.color_distance(avg_color, team_color)
                if dist < min_distance:
                    min_distance = dist
                    closest_team = team_name_key

            return closest_team, "unknown", avg_color
        else:
            # Return the K-means result
            return team_name, role, avg_color

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

        # Use ByteTrack for tracking - using default parameters to avoid compatibility issues
        self.tracker = sv.ByteTrack()

        # Store jersey numbers for players (to be detected automatically)
        self.jersey_numbers = {}

        # Store team information for players
        self.player_teams = {}
        self.team_classifier = TeamClassifier()

        # Store previous frame team color data for voting system
        self.team_color_buffer = {}  # {tracker_id: [(color1, frame_idx1), (color2, frame_idx2), ...]}
        self.analyzed_trackers = set()  # Trackers that have completed initial analysis

        # Initialize frame counter
        self.frame_count = 0

        # Initialize storage for baseline team detection
        self.baseline_frames_and_detections = []  # Store initial frames for team baseline calculation
        self.baseline_collection_complete = False  # Flag indicating if we've collected enough baseline data
        self.BASELINE_COLLECTION_FRAMES = 15  # Number of initial frames to collect for baseline

        # Store trajectories for visualization
        self.player_trajectories = defaultdict(lambda: deque(maxlen=30))
        self.puck_trajectory = deque(maxlen=15)

        # For puck interpolation - disabling for now
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
        # Define the number of frames to accumulate for voting
        TEAM_ANALYSIS_FRAMES = 15

        # Collect frames for baseline team analysis during first few frames
        if not self.baseline_collection_complete and self.frame_count <= self.BASELINE_COLLECTION_FRAMES:
            self.baseline_frames_and_detections.append((frame, tracked_detections))

            # If we've collected enough frames, initialize team baselines
            if self.frame_count == self.BASELINE_COLLECTION_FRAMES:
                # Initialize team baseline colors using K-means clustering
                success = self.team_classifier.initialize_team_baseline_colors(self.baseline_frames_and_detections)
                if success:
                    print("Team baseline colors established using K-means clustering.")
                    print(f"Team 1 color: {self.team_classifier.team_1_color}")
                    print(f"Team 2 color: {self.team_classifier.team_2_color}")
                else:
                    print("Could not establish team baseline colors from initial frames.")

                self.baseline_collection_complete = True
                self.baseline_frames_and_detections = []  # Clear memory

        if self.is_hockey_model:
            # Filter for hockey-specific classes that have jersey numbers
            # Loop through tracked_detections to process each detection
            for i, (xyxy, tracker_id) in enumerate(zip(tracked_detections.xyxy, tracked_detections.tracker_id)):
                # Get class_id for this specific detection
                class_id = tracked_detections.class_id[i] if i < len(tracked_detections.class_id) else None

                # Check if this tracker_id is already assigned - only assign new ones or still in analysis phase
                if tracker_id not in self.jersey_numbers:
                    # Classify based on class_id
                    if class_id == 4:  # Player class
                        # Assign jersey number based on existing numbers to avoid duplicates
                        assigned_number = len([k for k, v in self.jersey_numbers.items() if not v.startswith(('G', 'R'))]) + 1
                        self.jersey_numbers[tracker_id] = str(assigned_number)

                    elif class_id == 3:  # Goaltender class
                        self.jersey_numbers[tracker_id] = "G"

                    elif class_id == 6:  # Referee class
                        self.jersey_numbers[tracker_id] = "REF"

                # Check team assignment for players (not referees/goalies for team classification)
                if class_id == 4:  # Only for players
                    # Check if this tracker_id needs team analysis
                    if tracker_id not in self.player_teams:
                        # Initialize color buffer for this tracker
                        if tracker_id not in self.team_color_buffer:
                            self.team_color_buffer[tracker_id] = []

                        # Get the color for this frame
                        avg_color = self.team_classifier.calculate_average_color(frame, xyxy)

                        if avg_color is not None:
                            # Add color to buffer
                            self.team_color_buffer[tracker_id].append((avg_color, self.frame_count))

                            # If we have enough frames for analysis, determine final team assignment
                            if len(self.team_color_buffer[tracker_id]) >= TEAM_ANALYSIS_FRAMES:
                                # Calculate median color from buffer
                                r_values = [c[0][0] for c in self.team_color_buffer[tracker_id]]
                                g_values = [c[0][1] for c in self.team_color_buffer[tracker_id]]
                                b_values = [c[0][2] for c in self.team_color_buffer[tracker_id]]

                                median_r = sorted(r_values)[len(r_values)//2]
                                median_g = sorted(g_values)[len(g_values)//2]
                                median_b = sorted(b_values)[len(b_values)//2]

                                median_color = (median_r, median_g, median_b)

                                # Use the median color for team classification
                                team_name, jersey_color = self.team_classifier.classify_team(
                                    frame, xyxy, tracker_id, self.player_teams, class_id=4  # 4 is player class
                                )

                                # Update team classifier with the median color for this tracker
                                self.player_teams[tracker_id] = (team_name, jersey_color)
                                self.analyzed_trackers.add(tracker_id)

                                # Clear the buffer since analysis is complete
                                del self.team_color_buffer[tracker_id]
                    else:
                        # For trackers already analyzed, don't reclassify
                        pass

                elif class_id in [3, 6]:  # Goaltender or referee
                    # For goalies and referees, assign to special teams immediately
                    if tracker_id not in self.player_teams:
                        if class_id == 3:  # Goaltender
                            team_name, jersey_color = self.team_classifier.classify_team(
                                frame, xyxy, tracker_id, self.player_teams, class_id=3  # 3 is goalie class
                            )
                        elif class_id == 6:  # Referee
                            team_name, jersey_color = self.team_classifier.classify_team(
                                frame, xyxy, tracker_id, self.player_teams, class_id=6  # 6 is referee class
                            )
                        self.player_teams[tracker_id] = (team_name, jersey_color)

        else:
            # For general model, process each tracked detection
            for i, (xyxy, tracker_id) in enumerate(zip(tracked_detections.xyxy, tracked_detections.tracker_id)):
                # Check if this tracker_id is already assigned - only assign new ones
                if tracker_id not in self.jersey_numbers:
                    self.jersey_numbers[tracker_id] = str(len(self.jersey_numbers) + 1)

                # Classify team for general person detection
                if tracker_id not in self.player_teams:
                    # Initialize color buffer for this tracker
                    if tracker_id not in self.team_color_buffer:
                        self.team_color_buffer[tracker_id] = []

                    # Get the color for this frame
                    avg_color = self.team_classifier.calculate_average_color(frame, xyxy)

                    if avg_color is not None:
                        # Add color to buffer
                        self.team_color_buffer[tracker_id].append((avg_color, self.frame_count))

                        # If we have enough frames for analysis, determine final team assignment
                        if len(self.team_color_buffer[tracker_id]) >= TEAM_ANALYSIS_FRAMES:
                            # Calculate median color from buffer
                            r_values = [c[0][0] for c in self.team_color_buffer[tracker_id]]
                            g_values = [c[0][1] for c in self.team_color_buffer[tracker_id]]
                            b_values = [c[0][2] for c in self.team_color_buffer[tracker_id]]

                            median_r = sorted(r_values)[len(r_values)//2]
                            median_g = sorted(g_values)[len(g_values)//2]
                            median_b = sorted(b_values)[len(b_values)//2]

                            median_color = (median_r, median_g, median_b)

                            # Use the median color for team classification
                            team_name, jersey_color = self.team_classifier.classify_team(
                                frame, xyxy, tracker_id, self.player_teams
                            )

                            # Update team classifier with the median color for this tracker
                            self.player_teams[tracker_id] = (team_name, jersey_color)
                            self.analyzed_trackers.add(tracker_id)

                            # Clear the buffer since analysis is complete
                            del self.team_color_buffer[tracker_id]

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
        Currently disabled to improve stability
        """
        return None  # Disabled puck tracking

    def track(self, frame, frame_idx=0):
        """
        Process a frame with detection and tracking
        """
        # Increment frame counter for voting system
        self.frame_count += 1

        # Run detection
        detections = self.detect(frame)

        # Apply tracking
        tracked_detections = self.tracker.update_with_detections(detections)

        # Handle jersey number assignment and team classification on every frame for new trackers
        self.assign_jersey_numbers(frame, tracked_detections)

        # Get interpolated puck position - currently disabled
        puck_position = None

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