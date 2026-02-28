import cv2 as cv
import mediapipe as mp
import math
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """Calculate the angle at point b between points a-b-c"""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def get_angle_color(angle, optimal_range, warning_range):
    """Return color based on angle quality (green/yellow/red)"""
    if optimal_range[0] <= angle <= optimal_range[1]:
        return (0, 255, 0)  # Green - optimal
    elif warning_range[0] <= angle <= warning_range[1]:
        return (0, 255, 255)  # Yellow - acceptable
    else:
        return (0, 0, 255)  # Red - bad

def detect_rowing_phase(knee_angle, hip_angle, elbow_angle):
    """
    Detect the current rowing phase based on joint angles
    Returns: 'catch', 'drive', 'finish', or 'recovery'
    """
    # Catch: knees compressed (small angle), body compressed
    if knee_angle < 60 and hip_angle < 60:
        return 'catch'
    
    # Finish: legs extended (large knee angle), arms pulled in, body leaned back
    elif knee_angle > 140 and hip_angle > 140:
        return 'finish'
    
    # Drive: transitioning from catch, legs extending
    elif knee_angle < 140 and knee_angle >= 60:
        return 'drive'
    
    # Recovery: returning to catch position
    else:
        return 'recovery'

# Angle ranges for each rowing phase
ANGLE_RANGES = {
    'catch': {
        'knee': {'optimal': (35, 45), 'acceptable': (30, 55)},
        'hip': {'optimal': (35, 50), 'acceptable': (30, 60)},
        'ankle': {'optimal': (80, 100), 'acceptable': (70, 110)}
    },
    'drive': {
        'knee': {'optimal': (60, 120), 'acceptable': (50, 130)},
        'hip': {'optimal': (60, 120), 'acceptable': (50, 130)},
        'ankle': {'optimal': (85, 105), 'acceptable': (75, 115)}
    },
    'finish': {
        'knee': {'optimal': (160, 175), 'acceptable': (150, 180)},
        'hip': {'optimal': (145, 165), 'acceptable': (135, 175)},
        'ankle': {'optimal': (85, 105), 'acceptable': (75, 115)}
    },
    'recovery': {
        'knee': {'optimal': (100, 160), 'acceptable': (80, 170)},
        'hip': {'optimal': (100, 150), 'acceptable': (80, 160)},
        'ankle': {'optimal': (85, 105), 'acceptable': (75, 115)}
    }
}

# Phase colors for display
PHASE_COLORS = {
    'catch': (255, 100, 100),      # Light blue
    'drive': (100, 255, 100),      # Light green
    'finish': (100, 100, 255),     # Light red
    'recovery': (255, 255, 100)    # Light cyan
}

cap = cv.VideoCapture(r"C:\Users\User\Downloads\rowing_footage1.mov")

with mp_pose.Pose(static_image_mode=False,
                  model_complexity=1,
                  enable_segmentation=False,
                  min_detection_confidence=.5,
                  min_tracking_confidence=0.7) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        
        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        if results.pose_landmarks:
            h, w, _ = image.shape
            landmarks = results.pose_landmarks.landmark

            JOINTS = {
                'shoulder_l': 11, 'shoulder_r': 12,
                'elbow_l': 13, 'elbow_r': 14,
                'hip_l': 23, 'hip_r': 24,
                'knee_l': 25, 'knee_r': 26,
                'ankle_l': 27, 'ankle_r': 28,
                'wrist_l': 15, 'wrist_r': 16,
                'toe_l': 31, 'toe_r': 32
            }

            # Calculate angles
            hip_angle_l = calculate_angle(landmarks[JOINTS['shoulder_l']],
                                         landmarks[JOINTS['hip_l']], 
                                         landmarks[JOINTS['knee_l']])
            
            knee_angle_l = calculate_angle(landmarks[JOINTS['hip_l']], 
                                          landmarks[JOINTS['knee_l']], 
                                          landmarks[JOINTS['ankle_l']])
            
            ankle_angle_l = calculate_angle(landmarks[JOINTS['knee_l']], 
                                           landmarks[JOINTS['ankle_l']], 
                                           landmarks[JOINTS['toe_l']])
            
            elbow_angle_l = calculate_angle(landmarks[JOINTS['shoulder_l']],
                                           landmarks[JOINTS['elbow_l']],
                                           landmarks[JOINTS['wrist_l']])

            hip_angle_r = calculate_angle(landmarks[JOINTS['shoulder_r']], 
                                         landmarks[JOINTS['hip_r']], 
                                         landmarks[JOINTS['knee_r']])
            
            knee_angle_r = calculate_angle(landmarks[JOINTS['hip_r']], 
                                          landmarks[JOINTS['knee_r']], 
                                          landmarks[JOINTS['ankle_r']])
            
            ankle_angle_r = calculate_angle(landmarks[JOINTS['knee_r']], 
                                           landmarks[JOINTS['ankle_r']], 
                                           landmarks[JOINTS['toe_r']])

            # Detect rowing phase (using left side as primary)
            current_phase = detect_rowing_phase(knee_angle_l, hip_angle_l, elbow_angle_l)
            
            # Get angle ranges for current phase
            phase_ranges = ANGLE_RANGES[current_phase]

            # Colors based on angles and current phase
            knee_color_l = get_angle_color(knee_angle_l, 
                                            phase_ranges['knee']['optimal'],
                                            phase_ranges['knee']['acceptable'])

            hip_color_l = get_angle_color(hip_angle_l,
                                           phase_ranges['hip']['optimal'],
                                           phase_ranges['hip']['acceptable'])

            ankle_color_l = get_angle_color(ankle_angle_l,
                                             phase_ranges['ankle']['optimal'],
                                             phase_ranges['ankle']['acceptable'])

            knee_color_r = get_angle_color(knee_angle_r, 
                                            phase_ranges['knee']['optimal'],
                                            phase_ranges['knee']['acceptable'])

            hip_color_r = get_angle_color(hip_angle_r,
                                           phase_ranges['hip']['optimal'],
                                           phase_ranges['hip']['acceptable'])

            ankle_color_r = get_angle_color(ankle_angle_r,
                                             phase_ranges['ankle']['optimal'],
                                             phase_ranges['ankle']['acceptable'])

            # Draw joints
            for j in JOINTS.values():
                cx, cy = int(landmarks[j].x * w), int(landmarks[j].y * h)
                cv.circle(image, (cx, cy), 8, (0, 200, 200), 2)
                cv.circle(image, (cx, cy), 6, (0, 255, 255), -1)

            # Draw connections
            CONNECTIONS = [
                ('shoulder_l', 'elbow_l'), ('shoulder_r', 'elbow_r'),
                ('hip_l', 'knee_l'), ('hip_r', 'knee_r'),
                ('knee_l', 'ankle_l'), ('knee_r', 'ankle_r'),
                ('shoulder_l', 'shoulder_r'), ('hip_l', 'hip_r'),
                ('shoulder_l', 'hip_l'), ('shoulder_r', 'hip_r'),
                ('elbow_l', 'wrist_l'), ('elbow_r', 'wrist_r'),
                ('ankle_l', 'toe_l'), ('ankle_r', 'toe_r')
            ]

            for a, b in CONNECTIONS:
                ax, ay = int(landmarks[JOINTS[a]].x * w), int(landmarks[JOINTS[a]].y * h)
                bx, by = int(landmarks[JOINTS[b]].x * w), int(landmarks[JOINTS[b]].y * h)
                cv.line(image, (ax, ay), (bx, by), (255, 255, 255), 2)

            # Display phase at top of screen
            phase_text = f"PHASE: {current_phase.upper()}"
            cv.rectangle(image, (10, 10), (300, 60), (0, 0, 0), -1)
            cv.putText(image, phase_text, (20, 45),
                       cv.FONT_HERSHEY_SIMPLEX, 1.2, PHASE_COLORS[current_phase], 3)

            # Left side angles
            cv.putText(image, f"Hip: {int(hip_angle_l)}",
                       (int(landmarks[JOINTS['hip_l']].x * w) + 10, 
                        int(landmarks[JOINTS['hip_l']].y * h)), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, hip_color_l, 2)
            
            cv.putText(image, f"Knee: {int(knee_angle_l)}", 
                       (int(landmarks[JOINTS['knee_l']].x * w) + 10, 
                        int(landmarks[JOINTS['knee_l']].y * h)), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, knee_color_l, 2)
            
            cv.putText(image, f"Ankle: {int(ankle_angle_l)}", 
                       (int(landmarks[JOINTS['ankle_l']].x * w) + 10, 
                        int(landmarks[JOINTS['ankle_l']].y * h)), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, ankle_color_l, 2)

            # Right side angles
            cv.putText(image, f"Hip: {int(hip_angle_r)}",
                       (int(landmarks[JOINTS['hip_r']].x * w) - 80, 
                        int(landmarks[JOINTS['hip_r']].y * h)), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, hip_color_r, 2)
            
            cv.putText(image, f"Knee: {int(knee_angle_r)}", 
                       (int(landmarks[JOINTS['knee_r']].x * w) - 80, 
                        int(landmarks[JOINTS['knee_r']].y * h)), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, knee_color_r, 2)
            
            cv.putText(image, f"Ankle: {int(ankle_angle_r)}", 
                       (int(landmarks[JOINTS['ankle_r']].x * w) - 80, 
                        int(landmarks[JOINTS['ankle_r']].y * h)), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, ankle_color_r, 2)

            # Display phase-specific angle guidelines at bottom
            guidelines_y = h - 100
            cv.rectangle(image, (10, guidelines_y - 10), (w - 10, h - 10), (0, 0, 0), -1)
            
            cv.putText(image, f"{current_phase.upper()} Guidelines:", 
                       (20, guidelines_y + 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv.putText(image, f"Knee: {phase_ranges['knee']['optimal'][0]}-{phase_ranges['knee']['optimal'][1]}deg", 
                       (20, guidelines_y + 35),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv.putText(image, f"Hip: {phase_ranges['hip']['optimal'][0]}-{phase_ranges['hip']['optimal'][1]}deg", 
                       (200, guidelines_y + 35),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv.putText(image, f"Ankle: {phase_ranges['ankle']['optimal'][0]}-{phase_ranges['ankle']['optimal'][1]}deg", 
                       (380, guidelines_y + 35),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv.imshow('Rowing Pose Overlay', image)

        if cv.waitKey(7) & 0xFF == 27:
            break

cap.release()
cv.destroyAllWindows()