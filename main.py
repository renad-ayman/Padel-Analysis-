import sys
import os
import pickle
import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import read_video, save_video, measure_distance, get_center_of_bbox
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import constants  # Ensure the constants module is imported

# Load YOLOv8 model for detecting the ball and players
model = YOLO(r'models/best (1).pt')  # Use a pre-trained YOLO model or train your own on padel data

# Initialize scoreboard variables
team1_points = 0
team2_points = 0
team1_games = 0
team2_games = 0
team1_sets = 0
team2_sets = 0
current_serve = 'Team 1'  # Alternates between Team 1 and Team 2

# Court middle line (approximation based on video size; should be adjusted according to the actual video)
court_middle = 360  # Adjust this according to your video resolution

# Previous ball position
previous_ball_position = None

# Define score update logic
def update_score(team_scoring):
    global team1_points, team2_points, team1_games, team2_games, team1_sets, team2_sets

    if team_scoring == "Team 1":
        team1_points += 1
        if team1_points >= 6:  # Example scoring logic
            team1_games += 1
            team1_points = 0
    elif team_scoring == "Team 2":
        team2_points += 1
        if team2_points >= 6:
            team2_games += 1
            team2_points = 0

# Function to draw scoreboard on the video frame
def draw_scoreboard(frame):
    scoreboard = np.zeros((200, 600, 3), dtype=np.uint8)
    cv2.putText(scoreboard, f'Team 1', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(scoreboard, f'Team 2', (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(scoreboard, f'Points: {team1_points}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(scoreboard, f'Points: {team2_points}', (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(scoreboard, f'Games: {team1_games}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(scoreboard, f'Games: {team2_games}', (350, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(scoreboard, f'Sets: {team1_sets}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(scoreboard, f'Sets: {team2_sets}', (350, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Merge the scoreboard with the original frame
    frame[0:200, 0:600] = scoreboard
    return frame

def process_video(video):
    global previous_ball_position
    try:
        print("Starting script...")
        
        # Read video
        input_video_path = os.path.abspath(r'input_videos/input video.mp4')
        print(f"Reading video from {input_video_path}")
        video_frames = read_video(input_video_path)
        print(f"Read {len(video_frames)} frames from the video.")
        
        # Initialize trackers
        player_tracker = PlayerTracker(model_path="yolov8n")
        ball_tracker = BallTracker(model_path=os.path.abspath(r"models/best (1).pt"))
        print("Initialized trackers.")
        
        # Paths for stub files
        player_stub_path = os.path.abspath(r'stubs/player_detections.pkl')
        ball_stub_path = os.path.abspath(r'stubs/ball_detections.pkl')
        
        # Ensure the stubs directory exists
        os.makedirs(os.path.dirname(player_stub_path), exist_ok=True)
        os.makedirs(os.path.dirname(ball_stub_path), exist_ok=True)
        
        # Generate or read player detections
        if not os.path.exists(player_stub_path):
            print(f"Generating player detections and saving to {player_stub_path}")
            player_detections = player_tracker.detect_frames(video_frames, read_from_stub=False)
            with open(player_stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
            print(f"Player detections saved to {player_stub_path}.")
        else:
            print(f"Reading player detections from {player_stub_path}")
            with open(player_stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            print("Player detections read from stub.")
            
        # Generate or read ball detections
        if not os.path.exists(ball_stub_path):
            print(f"Generating ball detections and saving to {ball_stub_path}")
            ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=False)
            with open(ball_stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
            print(f"Ball detections saved to {ball_stub_path}.")
        else:
            print(f"Reading ball detections from {ball_stub_path}")
            with open(ball_stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            print("Ball detections read from stub.")
        
        ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
        print("Ball positions interpolated.")

        # Court Line Detector model
        court_model_path = os.path.abspath(r"models/keypoints_model.pth")
        court_line_detector = CourtLineDetector(court_model_path)
        print("Initialized court line detector.")
        court_keypoints = court_line_detector.predict(video_frames[0])
        print("Court keypoints predicted.")
        
        # Choose player
        print("Choosing and filtering players...")
        player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
        print("Players chosen and filtered.")

        # Mini court 
        print("Initializing mini court...")
        mini_court = MiniCourt(video_frames[0])
        print("Initialized mini court.")
        
        # Draw bboxes
        print("Drawing player bounding boxes...")
        output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
        print("Drawing ball bounding boxes...")
        output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
        print("Bounding boxes drawn.")
        
        # Draw mini court
        print("Drawing mini court on frames...")
        output_video_frames = mini_court.draw_mini_court(output_video_frames)
        print("Mini court drawn on frames.")
        
        # Process each frame for YOLO detection and scoring
        for i, frame in enumerate(output_video_frames):
            results = model(frame)
            for r in results:
                for detection in r.boxes:
                    cls = int(detection.cls[0])  # Convert to an integer if it's not already
                    label = model.names[cls]  # Get the class label name
                    

                    # Debugging: Print the detected class and confidence
                    print(f"Detected: {label}, Confidence: {detection.conf}")

                    if label == 'ball':  # Assuming YOLO is detecting the ball
                        current_ball_position = detection.xyxy[0]  # Get current ball position (bounding box top-left corner)

                        if previous_ball_position is not None:
                            # Example scoring logic: Check if the ball crossed the net
                            if previous_ball_position[1] < court_middle and current_ball_position[1] > court_middle:
                                print("Ball crossed the net from Team 1 to Team 2.")
                                update_score("Team 1")  # Update score based on ball crossing net from Team 1 side to Team 2
                            elif previous_ball_position[1] > court_middle and current_ball_position[1] < court_middle:
                                print("Ball crossed the net from Team 2 to Team 1.")
                                update_score("Team 2")

                        # Update previous position for next frame
                        previous_ball_position = current_ball_position 
        
        # Save the output video
        output_video_path = os.path.abspath(r'output_videos/output_video.mp4')
        print(f"Saving video to {output_video_path}")
        save_video(output_video_path, output_video_frames)
        print(f"Output video saved to {output_video_path}")
        
        return output_video_path

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "_main_":
    # Set the environment variable to avoid OpenMP error
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
   
iface = gr.Interface(
        fn=process_video,
        inputs=gr.Video(format="mp4"),
        outputs=gr.Video(format="mp4"),
        title="Padel Tennis Tracker",
    )
    
iface.launch(share=True)