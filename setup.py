#!/usr/bin/env python3
"""
Hockey Analytics System - Setup and Execution Script

This script helps set up and run the complete hockey analytics pipeline.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def install_requirements():
    """Install required Python packages"""
    print("Installing required packages...")
    req_file = Path("requirements.txt")

    if not req_file.exists():
        print(f"Error: {req_file} not found!")
        return False

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])
        print("Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False

def download_model():
    """Download the HockeyAI model from Hugging Face"""
    print("Downloading HockeyAI model...")
    try:
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        # Run the download script
        result = subprocess.run([sys.executable, "download_model.py"],
                              capture_output=True, text=True)

        if result.returncode == 0:
            print("Model downloaded successfully!")
            return True
        else:
            print(f"Error downloading model: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error running download script: {e}")
        return False

def run_analysis(video_path, model_path, output_path=None):
    """Run the hockey analytics on a video"""
    print(f"Running analysis on {video_path}...")

    cmd = [sys.executable, "main.py", "--video", video_path, "--model", model_path]
    if output_path:
        cmd.extend(["--output", output_path])

    try:
        subprocess.run(cmd)
        print("Analysis completed!")
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"Error running analysis: {e}")

def main():
    parser = argparse.ArgumentParser(description='Hockey Analytics System Setup and Execution')
    parser.add_argument('action', choices=['setup', 'run'],
                       help='Action to perform: setup (install deps and download model) or run (analyze video)')
    parser.add_argument('--video', type=str, help='Path to video file for analysis')
    parser.add_argument('--model', type=str, default='models/basic_yolo.pt',
                       help='Path to model file (default: models/basic_yolo.pt)')
    parser.add_argument('--output', type=str, help='Output video path (optional)')

    args = parser.parse_args()

    if args.action == 'setup':
        print("Setting up Hockey Analytics System...")
        success = install_requirements()
        if success:
            download_model()
        else:
            print("Failed to install requirements. Exiting.")
            sys.exit(1)

    elif args.action == 'run':
        if not args.video:
            print("Error: --video argument is required for run action")
            sys.exit(1)

        if not Path(args.video).exists():
            print(f"Error: Video file {args.video} does not exist")
            sys.exit(1)

        if not Path(args.model).exists():
            print(f"Error: Model file {args.model} does not exist")
            print("Did you run 'setup' first?")
            sys.exit(1)

        run_analysis(args.video, args.model, args.output)

if __name__ == "__main__":
    main()