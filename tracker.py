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

        # Variables for adaptive learning when baselines are not established
        self.potential_team_colors = []

    def extract_jersey_pixels(self, frame, bbox):
        """
        Extract pixels from the jersey area with improved black/white jersey detection
        Focus specifically on detecting truly black (0-50) and truly white (200-255) jersey colors
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
        h_start = int(player_region.shape[0] * 0.25)  # Start at 25% height
        h_end = int(player_region.shape[0] * 0.70)    # End at 70% height

        if h_start < h_end:
            jersey_area = player_region[h_start:h_end, :]

            # Convert to grayscale to measure brightness
            gray = cv2.cvtColor(jersey_area, cv2.COLOR_BGR2GRAY)

            # Identify regions that are likely to be black or white jerseys
            # Black jerseys: very dark regions (values 0-70)
            # White jerseys: very bright regions (values 180-255)
            # But be more inclusive to ensure we capture some pixels

            # Create masks for different brightness ranges
            black_mask = (gray >= 0) & (gray <= 80)  # Black jerseys (more inclusive)
            white_mask = (gray >= 170) & (gray <= 255)  # White jerseys (more inclusive)

            # Combine the masks to get pixels that are likely jersey colors
            jersey_mask = black_mask | white_mask

            # Get the pixels that match our jersey criteria
            jersey_pixels = jersey_area[jersey_mask]

            if len(jersey_pixels) > 0:
                return jersey_pixels.astype(np.float32)

            # If we still don't find enough distinctive pixels, return all pixels in the jersey area
            # but with a more relaxed approach
            pixels = jersey_area.reshape(-1, 3)

            if len(pixels) > 0:
                # Prioritize pixels that have some contrast to distinguish from ice
                pixel_brightness = np.mean(pixels, axis=1)

                # Be more inclusive - exclude only the extremes (likely ice or shadows)
                jersey_brightness_mask = (pixel_brightness >= 10) & (pixel_brightness <= 245)
                filtered_pixels = pixels[jersey_brightness_mask]

                if len(filtered_pixels) > 0:
                    return filtered_pixels.astype(np.float32)
                else:
                    # Last resort: return all pixels if none meet the criteria
                    return pixels.astype(np.float32)

        # Final fallback: return all pixels in the original area if shape constraints prevented extraction
        jersey_area_full = player_region[int(player_region.shape[0] * 0.25):int(player_region.shape[0] * 0.70), :]
        if jersey_area_full.size > 0:
            pixels = jersey_area_full.reshape(-1, 3)
            return pixels.astype(np.float32)

        return None

    def extract_jersey_contour_pixels(self, frame, bbox):
        """
        Step 1: Extract jersey pixels using contour detection.
        Isolates the player from the ice background using Otsu's thresholding,
        finds the player contour, and extracts the torso region.
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
        player_crop = frame[y1:y2, x1:x2]

        # Convert to grayscale
        gray = cv2.cvtColor(player_crop, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's thresholding to separate the player from the bright ice
        # THRESH_BINARY_INV makes the darker player white (255) and the bright ice black (0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # Fallback if no contours found: take a simple center crop
            h, w = player_crop.shape[:2]
            fallback_crop = player_crop[int(h*0.2):int(h*0.6), int(w*0.2):int(w*0.8)]
            return fallback_crop.reshape(-1, 3).astype(np.float32)

        # Get the largest contour by area (this should be the player)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get bounding rectangle of the contour itself
        cx, cy, cw, ch = cv2.boundingRect(largest_contour)

        # Define the jersey region vertically (e.g., top 20% to 60% of the player contour)
        # This avoids the helmet (top 0-20%) and pants/skates (bottom 60-100%)
        jersey_top = cy + int(ch * 0.2)
        jersey_bottom = cy + int(ch * 0.6)

        # Create a mask for the player's contour shape
        contour_mask = np.zeros(player_crop.shape[:2], dtype=np.uint8)
        cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)

        # Create a mask for just the torso height
        torso_height_mask = np.zeros(player_crop.shape[:2], dtype=np.uint8)
        torso_height_mask[jersey_top:jersey_bottom, cx:cx+cw] = 255

        # Combine masks: Must be inside the player's contour AND inside the torso height
        final_mask = cv2.bitwise_and(contour_mask, torso_height_mask)

        # Extract the original RGB pixels using the combined mask
        pixels = player_crop[final_mask == 255]

        if len(pixels) == 0:
            # Fallback if the mask ended up empty
            h, w = player_crop.shape[:2]
            fallback_crop = player_crop[int(h*0.2):int(h*0.6), int(w*0.2):int(w*0.8)]
            return fallback_crop.reshape(-1, 3).astype(np.float32)

        return pixels.astype(np.float32)

    def initialize_team_baseline_colors(self, frames_and_detections):
        """
        Step 2: Find Team Baselines using K-Means Clustering
        Uses the new contour-extracted pixels. Filters removed to allow pure black/white.
        """
        from sklearn.cluster import KMeans

        all_torso_pixels = []

        # Collect jersey pixels from PLAYER detections only (class_id 4)
        # Excludes refs, goalies, rink features which contaminate baselines
        for frame, detections in frames_and_detections:
            for i, xyxy in enumerate(detections.xyxy):
                # Filter: only use player detections (class_id == 4) if class info available
                if detections.class_id is not None and len(detections.class_id) > i:
                    if detections.class_id[i] != 4:
                        continue
                pixels = self.extract_jersey_contour_pixels(frame, xyxy)
                if pixels is not None and len(pixels) > 0:
                    # Randomly sample pixels if the contour is large to speed up KMeans
                    if len(pixels) > 150:
                        indices = np.random.choice(len(pixels), 150, replace=False)
                        pixels = pixels[indices]
                    all_torso_pixels.append(pixels)

        if len(all_torso_pixels) == 0:
            return False

        # Concatenate all pixels from all detections
        all_pixels = np.vstack(all_torso_pixels)

        if len(all_pixels) < 2:
            return False

        # Apply K-means clustering with k=2 to find team baseline colors
        n_clusters = min(2, len(all_pixels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(all_pixels)

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
        Uses contour-extracted pixels to find the dominant color via median,
        then compares to baselines using HSV distance (robust to lighting changes).
        """
        # Step 1: Extract true jersey pixels using contour masking
        pixels = self.extract_jersey_contour_pixels(frame, bbox)

        if pixels is None or len(pixels) < 1:
            return "unknown", "unknown", (128, 128, 128)

        # Step 2: Find dominant color using median (cheaper & more stable than KMeans k=1)
        player_dominant_color = np.median(pixels, axis=0).astype(int)

        # If baselines aren't initialized yet, use adaptive approach
        if not self.kmeans_initialized:
            self.potential_team_colors.append(player_dominant_color)

            if len(self.potential_team_colors) >= 10:
                brightness_values = [np.mean(color) for color in self.potential_team_colors]
                sorted_indices = np.argsort(brightness_values)
                darkest_idx = sorted_indices[0]
                brightest_idx = sorted_indices[-1]
                self.team_1_color = self.potential_team_colors[darkest_idx].astype(int)
                self.team_2_color = self.potential_team_colors[brightest_idx].astype(int)
                self.kmeans_initialized = True
                print(f"Adaptive baselines established: Team 1 {self.team_1_color}, Team 2 {self.team_2_color}")

            if self.kmeans_initialized:
                # Compare using HSV distance
                dist_to_team1 = self.hsv_color_distance(player_dominant_color, self.team_1_color if self.team_1_color is not None else np.array([128, 128, 128]))
                dist_to_team2 = self.hsv_color_distance(player_dominant_color, self.team_2_color if self.team_2_color is not None else np.array([128, 128, 128]))

                if dist_to_team1 <= dist_to_team2:
                    return "team_1", "jersey_color_1", tuple(player_dominant_color)
                else:
                    return "team_2", "jersey_color_2", tuple(player_dominant_color)
            else:
                brightness = np.mean(player_dominant_color)
                if brightness > 150:
                    return "team_2", "light_jersey", tuple(player_dominant_color)
                elif brightness < 100:
                    return "team_1", "dark_jersey", tuple(player_dominant_color)
                else:
                    return "unknown", "medium_jersey", tuple(player_dominant_color)
        else:
            # Compare using HSV distance (robust to lighting changes)
            dist_to_team1 = self.hsv_color_distance(player_dominant_color, self.team_1_color if self.team_1_color is not None else np.array([128, 128, 128]))
            dist_to_team2 = self.hsv_color_distance(player_dominant_color, self.team_2_color if self.team_2_color is not None else np.array([128, 128, 128]))

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

    def bgr_to_lab(self, bgr):
        """
        Convert BGR color (OpenCV native format) to LAB color space
        """
        bgr_arr = np.uint8([[list(bgr)]])
        lab = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2LAB)
        return tuple(lab[0, 0])

    def lab_color_distance(self, lab1, lab2):
        """
        Calculate distance between two colors in LAB space using Euclidean distance
        """
        dL = lab1[0] - lab2[0]
        da = lab1[1] - lab2[1]
        db = lab1[2] - lab2[2]
        return math.sqrt(dL*dL + da*da + db*db)

    def bgr_to_hsv(self, bgr):
        """
        Convert BGR color (OpenCV native format) to HSV
        """
        bgr_arr = np.uint8([[list(bgr)]])
        hsv = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2HSV)
        return tuple(hsv[0, 0])

    def hsv_color_distance(self, bgr1, bgr2):
        """
        Compare two BGR colors via HSV distance.
        Hue is weighted heavily but ignored when saturation is very low
        (black/white jerseys), in which case brightness (value) decides.
        """
        h1, s1, v1 = self.bgr_to_hsv(bgr1)
        h2, s2, v2 = self.bgr_to_hsv(bgr2)

        # Circular hue distance (OpenCV hue range 0-180)
        hue_diff = min(abs(int(h1) - int(h2)), 180 - abs(int(h1) - int(h2)))

        # If both colors have very low saturation, hue is meaningless
        # Use brightness (value) to distinguish black vs white
        if s1 < 40 and s2 < 40:
            return abs(int(v1) - int(v2))

        return hue_diff * 2.0 + abs(int(s1) - int(s2)) * 0.5 + abs(int(v1) - int(v2)) * 0.3

    def color_distance(self, color1, color2):
        """
        Calculate distance between two BGR colors in HSV space (legacy wrapper)
        """
        return self.hsv_color_distance(color1, color2)

    def classify_player_by_color(self, frame, bbox, tracker_id):
        """
        Classify a player based on their jersey color using K-means approach with improved black/white detection
        """
        # Use the new K-means based team assignment with improved extraction
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

        # Use ByteTrack for tracking - increase lost_track_buffer so tracks survive brief occlusions
        self.tracker = sv.ByteTrack(lost_track_buffer=60)

        # Store jersey numbers for players (to be detected automatically)
        self.jersey_numbers = {}

        # Store team information for players
        self.player_teams = {}
        self.team_classifier = TeamClassifier()

        # Store previous frame team color data for voting system
        self.team_color_buffer = {}  # {tracker_id: [(color1, frame_idx1), (color2, frame_idx2), ...]}
        self.analyzed_trackers = set()  # Trackers that have completed initial analysis

        # Team validation: periodic re-check to correct wrong initial assignments
        # Requires strong evidence (>80% disagreement) to flip — prevents lighting-induced flickering
        self.team_validation_buffer = {}  # {tracker_id: deque of team votes}
        self.TEAM_VALIDATION_WINDOW = 30  # Number of validation samples before considering a flip
        self.TEAM_FLIP_THRESHOLD = 0.8    # Fraction of disagreeing votes needed to flip team

        # Initialize frame counter
        self.frame_count = 0

        # Initialize storage for baseline team detection
        self.baseline_frames_and_detections = []  # Store initial frames for team baseline calculation
        self.baseline_collection_complete = False  # Flag indicating if we've collected enough baseline data
        self.BASELINE_COLLECTION_FRAMES = 15  # Number of initial frames to collect for baseline

        # Store trajectories for visualization
        self.player_trajectories = defaultdict(lambda: deque(maxlen=30))
        self.puck_trajectory = deque(maxlen=15)

        # Re-ID support: store recently lost trackers {tracker_id: {'last_pos': (x,y), 'last_color': (r,g,b), 'timestamp': frame_idx}}
        self.lost_trackers = {}
        self.REID_PROXIMITY_THRESHOLD = 150  # Pixels (increased to catch fast-moving players)
        self.REID_COLOR_THRESHOLD = 40       # LAB color distance threshold for appearance matching
        self.REID_MAX_AGE = 90               # Frames (increased to handle longer occlusions)

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
        if not self.baseline_collection_complete:
            self.baseline_frames_and_detections.append((frame, tracked_detections))

            # Check if we've collected enough frames OR if we have sufficient players detected
            # Initialize team baselines when we reach the required frame count
            if self.frame_count >= self.BASELINE_COLLECTION_FRAMES:
                # Check if we have any player detections in our collected frames
                has_players = any(len(detections.xyxy) > 0 for _, detections in self.baseline_frames_and_detections)

                if has_players:
                    # Initialize team baseline colors using K-means clustering
                    success = self.team_classifier.initialize_team_baseline_colors(self.baseline_frames_and_detections)
                    if success:
                        print("Team baseline colors established using K-means clustering.")
                        print(f"Team 1 color: {self.team_classifier.team_1_color}")
                        print(f"Team 2 color: {self.team_classifier.team_2_color}")

                        # Verify that both team colors were set (not None)
                        if self.team_classifier.team_1_color is not None and self.team_classifier.team_2_color is not None:
                            print("Both team baselines successfully established!")
                        else:
                            print("WARNING: One or both team baselines are None")
                    else:
                        print("Could not establish team baseline colors from initial frames. Using fallback.")

                        # Fallback: create rough baselines based on common hockey team colors
                        # Team 1: Darker colors (black/blue) Team 2: Lighter colors (white/red)
                        self.team_classifier.team_1_color = np.array([40, 40, 40])  # Dark
                        self.team_classifier.team_2_color = np.array([200, 200, 200])  # Light
                        self.team_classifier.kmeans_initialized = True
                        print(f"Fallback team baselines created: Team 1 {self.team_classifier.team_1_color}, Team 2 {self.team_classifier.team_2_color}")
                else:
                    print("No players detected in initial frames. Using fallback team colors.")
                    # Fallback: create rough baselines based on common hockey team colors
                    self.team_classifier.team_1_color = np.array([40, 40, 40])  # Dark
                    self.team_classifier.team_2_color = np.array([200, 200, 200])  # Light
                    self.team_classifier.kmeans_initialized = True
                    print(f"Fallback team baselines created: Team 1 {self.team_classifier.team_1_color}, Team 2 {self.team_classifier.team_2_color}")

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
                    if tracker_id not in self.player_teams:
                        # Use vote-based assignment: classify each frame, assign by majority
                        if tracker_id not in self.team_color_buffer:
                            self.team_color_buffer[tracker_id] = []

                        if self.baseline_collection_complete and self.team_classifier.kmeans_initialized:
                            # Baselines ready: use robust contour + LAB assignment
                            team_name, role, avg_color = self.team_classifier.assign_player_to_team(frame, xyxy)
                            if team_name in ('team_1', 'team_2'):
                                self.team_color_buffer[tracker_id].append((team_name, avg_color))

                            # Assign after 5 votes via majority
                            votes = self.team_color_buffer.get(tracker_id, [])
                            if len(votes) >= 5:
                                team_votes = [v[0] for v in votes]
                                t1_count = team_votes.count('team_1')
                                t2_count = team_votes.count('team_2')
                                winner = 'team_1' if t1_count >= t2_count else 'team_2'
                                last_color = votes[-1][1] if votes[-1][1] else (128, 128, 128)
                                self.player_teams[tracker_id] = (winner, last_color)
                                self.analyzed_trackers.add(tracker_id)
                                if tracker_id in self.team_color_buffer:
                                    del self.team_color_buffer[tracker_id]
                        else:
                            # Baselines not ready: buffer raw colors and classify at end
                            avg_color = self.team_classifier.calculate_average_color(frame, xyxy)
                            if avg_color is not None:
                                self.team_color_buffer[tracker_id].append(avg_color)
                                if len(self.team_color_buffer[tracker_id]) >= TEAM_ANALYSIS_FRAMES:
                                    team_name, jersey_color = self.team_classifier.classify_team(
                                        frame, xyxy, tracker_id, self.player_teams, class_id=4
                                    )
                                    self.player_teams[tracker_id] = (team_name, jersey_color)
                                    self.analyzed_trackers.add(tracker_id)
                                    del self.team_color_buffer[tracker_id]
                    else:
                        # Already assigned: periodic re-validation (every 5 frames)
                        if self.frame_count % 5 == 0 and self.team_classifier.kmeans_initialized:
                            if tracker_id not in self.team_validation_buffer:
                                self.team_validation_buffer[tracker_id] = deque(maxlen=self.TEAM_VALIDATION_WINDOW)

                            team_name, role, avg_color = self.team_classifier.assign_player_to_team(frame, xyxy)
                            if team_name in ('team_1', 'team_2'):
                                self.team_validation_buffer[tracker_id].append(team_name)

                            # Only flip if strong evidence of wrong assignment
                            buf = self.team_validation_buffer.get(tracker_id)
                            if buf and len(buf) >= self.TEAM_VALIDATION_WINDOW:
                                current_team = self.player_teams[tracker_id][0]
                                if current_team in ('team_1', 'team_2'):
                                    disagree_count = sum(1 for v in buf if v != current_team)
                                    if disagree_count / len(buf) >= self.TEAM_FLIP_THRESHOLD:
                                        other_team = 'team_2' if current_team == 'team_1' else 'team_1'
                                        self.player_teams[tracker_id] = (other_team, self.player_teams[tracker_id][1])
                                        buf.clear()

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
                    if tracker_id not in self.team_color_buffer:
                        self.team_color_buffer[tracker_id] = []

                    if self.baseline_collection_complete and self.team_classifier.kmeans_initialized:
                        # Baselines ready: vote-based assignment
                        team_name, role, avg_color = self.team_classifier.assign_player_to_team(frame, xyxy)
                        if team_name in ('team_1', 'team_2'):
                            self.team_color_buffer[tracker_id].append((team_name, avg_color))

                        votes = self.team_color_buffer.get(tracker_id, [])
                        if len(votes) >= 5:
                            team_votes = [v[0] for v in votes]
                            t1_count = team_votes.count('team_1')
                            t2_count = team_votes.count('team_2')
                            winner = 'team_1' if t1_count >= t2_count else 'team_2'
                            last_color = votes[-1][1] if votes[-1][1] else (128, 128, 128)
                            self.player_teams[tracker_id] = (winner, last_color)
                            self.analyzed_trackers.add(tracker_id)
                            if tracker_id in self.team_color_buffer:
                                del self.team_color_buffer[tracker_id]
                    else:
                        # Baselines not ready: buffer raw colors and classify at end
                        avg_color = self.team_classifier.calculate_average_color(frame, xyxy)
                        if avg_color is not None:
                            self.team_color_buffer[tracker_id].append(avg_color)
                            if len(self.team_color_buffer[tracker_id]) >= TEAM_ANALYSIS_FRAMES:
                                team_name, jersey_color = self.team_classifier.classify_team(
                                    frame, xyxy, tracker_id, self.player_teams
                                )
                                self.player_teams[tracker_id] = (team_name, jersey_color)
                                self.analyzed_trackers.add(tracker_id)
                                del self.team_color_buffer[tracker_id]
                else:
                    # Already assigned: periodic re-validation
                    if self.frame_count % 5 == 0 and self.team_classifier.kmeans_initialized:
                        if tracker_id not in self.team_validation_buffer:
                            self.team_validation_buffer[tracker_id] = deque(maxlen=self.TEAM_VALIDATION_WINDOW)

                        team_name, role, avg_color = self.team_classifier.assign_player_to_team(frame, xyxy)
                        if team_name in ('team_1', 'team_2'):
                            self.team_validation_buffer[tracker_id].append(team_name)

                        buf = self.team_validation_buffer.get(tracker_id)
                        if buf and len(buf) >= self.TEAM_VALIDATION_WINDOW:
                            current_team = self.player_teams[tracker_id][0]
                            if current_team in ('team_1', 'team_2'):
                                disagree_count = sum(1 for v in buf if v != current_team)
                                if disagree_count / len(buf) >= self.TEAM_FLIP_THRESHOLD:
                                    other_team = 'team_2' if current_team == 'team_1' else 'team_1'
                                    self.player_teams[tracker_id] = (other_team, self.player_teams[tracker_id][1])
                                    buf.clear()

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

        # RE-ID: Detect lost trackers and add to lost_trackers (with appearance info)
        current_trackers = set(tracked_detections.tracker_id)
        all_known_trackers = set(self.player_trajectories.keys()) | set(self.lost_trackers.keys())
        newly_lost = all_known_trackers - current_trackers - set(self.jersey_numbers.keys())

        for lost_id in newly_lost:
            if lost_id in self.player_trajectories and len(self.player_trajectories[lost_id]) > 0:
                last_pos = self.player_trajectories[lost_id][-1]
                # Store appearance (jersey color) for color-aware re-ID
                last_color = None
                team_info = self.player_teams.get(lost_id)
                if team_info and isinstance(team_info, tuple) and len(team_info) >= 2:
                    jersey_color = team_info[1]
                    if isinstance(jersey_color, tuple) and len(jersey_color) == 3:
                        last_color = jersey_color
                self.lost_trackers[lost_id] = {
                    'last_pos': last_pos,
                    'last_color': last_color,
                    'timestamp': self.frame_count
                }

        # Handle jersey number assignment and team classification on every frame for new trackers
        self.assign_jersey_numbers(frame, tracked_detections)

        # RE-ID: Check for lost trackers and attempt to re-link them if a new tracker is nearby
        for lost_id in list(self.lost_trackers.keys()):
            if lost_id not in current_trackers:
                if self.frame_count - self.lost_trackers[lost_id]['timestamp'] > self.REID_MAX_AGE:
                    del self.lost_trackers[lost_id]

        for i, (xyxy, tracker_id) in enumerate(zip(tracked_detections.xyxy, tracked_detections.tracker_id)):
            center_point = (int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2))
            if tracker_id not in self.player_trajectories or len(self.player_trajectories[tracker_id]) == 0:
                best_match_id = None
                best_match_score = float('inf')

                for lost_id, data in list(self.lost_trackers.items()):
                    last_pos = data['last_pos']
                    dist = math.sqrt((center_point[0] - last_pos[0])**2 + (center_point[1] - last_pos[1])**2)
                    if dist >= self.REID_PROXIMITY_THRESHOLD:
                        continue

                    # If lost tracker has color info, also check appearance similarity
                    color_score = 0
                    lost_color = data.get('last_color')
                    if lost_color is not None:
                        new_color = self.team_classifier.calculate_average_color(frame, xyxy)
                        if new_color is not None:
                            lost_lab = self.team_classifier.bgr_to_lab(lost_color)
                            new_lab = self.team_classifier.bgr_to_lab(new_color)
                            color_score = self.team_classifier.lab_color_distance(lost_lab, new_lab)
                            if color_score > self.REID_COLOR_THRESHOLD:
                                continue  # Colors too different, skip this candidate

                    # Combined score: spatial distance + color distance
                    combined_score = dist + color_score
                    if combined_score < best_match_score:
                        best_match_score = combined_score
                        best_match_id = lost_id

                if best_match_id is not None:
                    # Transfer full state from lost tracker to new tracker
                    self.player_trajectories[tracker_id] = deque(self.player_trajectories.get(best_match_id, deque(maxlen=30)), maxlen=30)
                    if best_match_id in self.jersey_numbers:
                        self.jersey_numbers[tracker_id] = self.jersey_numbers[best_match_id]
                    if best_match_id in self.player_teams:
                        self.player_teams[tracker_id] = self.player_teams[best_match_id]
                    if best_match_id in self.team_color_buffer:
                        self.team_color_buffer[tracker_id] = self.team_color_buffer[best_match_id]
                        del self.team_color_buffer[best_match_id]
                    if best_match_id in self.analyzed_trackers:
                        self.analyzed_trackers.add(tracker_id)
                    del self.lost_trackers[best_match_id]

            # Store trajectory for tracked objects (players, goaltenders, refs)
            if self.is_hockey_model:
                if i < len(tracked_detections.class_id):
                    class_id = tracked_detections.class_id[i]
                    if class_id in [3, 4, 6]:
                        self.player_trajectories[tracker_id].append(center_point)
            else:
                self.player_trajectories[tracker_id].append(center_point)

        # Get interpolated puck position - currently disabled
        puck_position = None

        if puck_position is not None:
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