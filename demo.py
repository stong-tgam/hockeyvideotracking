import cv2
import numpy as np
from tracker import HockeyTracker
from homography import HockeyRinkHomography
from analytics import HockeyAnalytics
from visualization import HockeyVisualization
import argparse
import os

def demo_main(video_path, model_path, output_path=None):
    """
    Demo version of the hockey analytics pipeline that doesn't require user input
    """
    print("Initializing Hockey Analytics System (Demo Mode)...")

    # Initialize components
    tracker = HockeyTracker(model_path)
    homography_calc = HockeyRinkHomography()
    analytics = HockeyAnalytics(homography_calc)

    # Create visualization instance
    visualization = HockeyVisualization()

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")

    # Setup output video writer if path provided
    if output_path:
        # Try multiple codecs for better compatibility
        codecs_to_try = [
            cv2.VideoWriter_fourcc(*'mp4v'),  # MP4
            cv2.VideoWriter_fourcc(*'XVID'),  # AVI
            cv2.VideoWriter_fourcc(*'MJPG'),  # Motion JPEG
            cv2.VideoWriter_fourcc(*'X264'),  # H.264 (if available)
        ]

        out = None
        # Ensure fps is valid
        actual_fps = max(1.0, float(fps)) if fps > 0 else 30.0
        for fourcc in codecs_to_try:
            out = cv2.VideoWriter(output_path, fourcc, actual_fps, (width, height))
            # Test if we can actually write to this VideoWriter
            if out.isOpened():
                test_frame = np.zeros((height, width, 3), dtype=np.uint8)
                out.write(test_frame)
                # Check if write was successful by verifying the writer is still open
                if out.isOpened():
                    out.release()
                    # Re-open for actual use
                    out = cv2.VideoWriter(output_path, fourcc, actual_fps, (width, height))
                    print(f"Successfully opened video writer with codec")
                    break
                else:
                    out.release()
                    out = None
            else:
                out.release()
                out = None

        if out is None or not out.isOpened():
            print(f"Warning: Could not create video writer for {output_path}. Will run without saving.")
            out = None

    frame_idx = 0
    selected_player_for_heatmap = None

    print("Starting analysis... (Auto mode - no user input required)")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        print(f"Processing frame {frame_idx}/{total_frames}", end='\r')

        # Run tracking
        tracked_detections, puck_position = tracker.track(frame, frame_idx)

        # Update homography if possible
        homography_calc.initialize_if_needed(tracked_detections)

        # Compute homography matrix for this frame
        homography_calc.compute_homography(tracked_detections)

        # Process analytics
        # For this implementation, we'll focus on tracking and team classification without detailed analytics
        # We'll just return a placeholder rink visualization
        possessed_player_id = None

        # Calculate puck possession
        possessed_player_id = analytics.calculate_puck_possession(analytics.player_positions, puck_position, frame_idx)

        # Create a placeholder for the analytics (since we're not showing rink view)
        rink_vis = np.zeros((100, 100, 3), dtype=np.uint8)  # Small placeholder

        # Visualize tracking on original frame
        annotated_frame = visualization.create_single_view(frame, tracked_detections, tracker)

        # Combine original view with rink view
        # Since we're not showing the rink view anymore, we'll just use the annotated frame
        combined_frame = annotated_frame

        # Display frame
        cv2.imshow('Hockey Analytics - Original + Rink View (Press q to quit)', combined_frame)

        # Handle user input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Select player for heatmap visualization
            if tracker.jersey_numbers:
                print("\nAvailable players:", list(tracker.jersey_numbers.values()))
                print("Selecting first player for heatmap demonstration")
                if tracker.jersey_numbers:
                    first_tracker_id = next(iter(tracker.jersey_numbers.keys()))
                    selected_player_for_heatmap = first_tracker_id
                    print(f"Showing heatmap for player #{tracker.jersey_numbers[first_tracker_id]}")
            else:
                print("No players registered yet")

        # Write frame if output path provided
        if output_path:
            out.write(combined_frame)

        frame_idx += 1

    # End of video reached - process all frames

    # Cleanup
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

    # Export analytics to CSV
    analytics.export_csv_data("player_analytics.csv")

    # Print final statistics
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE (Demo Mode)")
    print("="*50)

    print("\nPlayer Statistics:")
    for tracker_id, distance in analytics.player_distances.items():
        jersey_num = tracker.jersey_numbers.get(tracker_id, tracker_id)
        print(f"Player #{jersey_num} (ID {tracker_id}): {distance:.2f} feet traveled")

    if analytics.puck_control_history:
        print("\nPuck Possession:")
        for player_id, data in analytics.puck_control_history.items():
            jersey_num = tracker.jersey_numbers.get(player_id, player_id)
            frames_in_possession = len(data['frames'])
            print(f"Player #{jersey_num} (ID {player_id}): {frames_in_possession} frames in possession")


def get_jersey_number(self, tracker_id):
    """
    Helper method to get jersey number for a tracker ID
    """
    return self.jersey_numbers.get(tracker_id, tracker_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hockey Analytics System Demo')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model file')
    parser.add_argument('--output', type=str, help='Path to output video file (optional)')

    args = parser.parse_args()

    # Add helper method to tracker class
    HockeyTracker.get_jersey_number = get_jersey_number

    if not os.path.exists(args.video):
        print(f"Error: Video file {args.video} does not exist")
        exit(1)

    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} does not exist")
        exit(1)

    demo_main(args.video, args.model, args.output)