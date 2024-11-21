# Padel Tennis Tracker
![padel court](https://github.com/renad-ayman/Padel-Analysis-.git)

## Introduction

The Padel Tennis Tracker is a system designed to track players and ball movements in a padel game using computer vision techniques. The tracker uses machine learning models for detecting and tracking the ball and players, and outputs annotated video frames for visualization.

## Installation

To use the Padel Tennis Tracker, follow these steps:

1. Clone this repository:
   bash
   git clone https://github.com/your-repo/padel-tennis-tracker.git
   cd padel-tennis-tracker
   2. Install dependencies:
   bash
   pip install -r requirements.txt
   

## Usage

### Ball Tracker

The Ball Tracker class is responsible for detecting the ball and interpolating its position when it's not detected. It also identifies frames where the ball is shot based on changes in its vertical position.

### Player Tracker

The Player Tracker class tracks the movement of players on the court. It uses a YOLO-based model to detect players in each frame of the video. The system then filters and selects the players based on their proximity to specific court keypoints.
### Mini Court Tracker
The Mini Court Tracker provides a visual representation of a scaled-down version of the Padel court within the video frame. This helps in tracking and analyzing player movement relative to key court features like the service lines, middle line, and net. The mini-court is drawn using pixel-based coordinates that are scaled from real-world dimensions.

### Key Features of the Mini Court
Service lines: Marking the service areas on both sides of the court.

Middle line: Divides the court into two halves.

Net: Positioned at the center of the court.

#### Court corners: Representing the boundaries of the playing area.
---

### Notes
- The system relies on the YOLO model for object detection and tracking.
- Ball position interpolation is performed using pandas, ensuring smooth movement even when the ball is temporarily out of frame.

---
