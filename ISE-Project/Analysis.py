import cv2 as cv
import mediapipe as mp
import math
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

optimal_count = 0
acceptable_count = 0
bad_count = 0 



def calculate_angle(a, b, c):
    """Calculate the angle at point b between points a-b-c"""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)) #bit o maths jargon doc product rule quite confusing tbh
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)   #spits out angle

def get_angle_color(angle, optimal_range, warning_range):
    """Return color based on angle quality (green/yellow/red)"""
    if optimal_range[0] <= angle <= optimal_range[1]:
        return (0, 255, 0)  # Green - optimal
    elif warning_range[0] <= angle <= warning_range[1]:
        return (0, 255, 255)  # Yellow - acceptable
    else:
        return (0, 0, 255)  # Red - bad
    

def color_to_quality(color):
    color = tuple(color)
    if color == (0, 255, 0):
        return "optimal"
    elif color == (0, 255, 255):
        return "acceptable"
    else:
        return "bad"





# Angle ranges for each rowing phase
ANGLE_RANGES = {
    'catch': {
        'knee': {'optimal': (35, 45), 'acceptable': (30, 55)},
        'hip': {'optimal': (35, 50), 'acceptable': (30, 60)},
        'ankle': {'optimal': (75, 100), 'acceptable': (70, 110)}
    },
    'drive': {
        'knee': {'optimal': (60, 120), 'acceptable': (50, 130)},
        'hip': {'optimal': (60, 120), 'acceptable': (50, 130)},
        'ankle': {'optimal': (85, 105), 'acceptable': (75, 115)}
    },
    'finish': {
        'knee': {'optimal': (163, 175), 'acceptable': (150, 180)},
        'hip': {'optimal': (120, 145), 'acceptable': (110, 150)},  
        'ankle': {'optimal': (85, 105), 'acceptable': (75, 115)}
    },
    'recovery': {
        'knee': {'optimal': (120, 160), 'acceptable': (80, 170)},
        'hip': {'optimal': (100, 150), 'acceptable': (80, 160)},
        'ankle': {'optimal': (85, 105), 'acceptable': (75, 115)}
    }
}

# Phase colors for display 
PHASE_COLORS = {
    'catch': (255, 100, 100),      #  blue
    'drive': (100, 255, 100),      #  green
    'finish': (100, 100, 255),     #  red
    'recovery': (255, 255, 100)    #  cyan
}

class StrokeMachine:
    def __init__(self):
        self.state = "recovery"
        self.pending = None
        self.count = 0

        self.debounce = {
            "finish":   3,
            "recovery": 4,
            "catch":    3,
            "drive":    4
        }

        # each state can only go to one next state — enforces the stroke cycle
        self.transitions = {
            "finish":   ["recovery"],
            "recovery": ["catch"],
            "catch":    ["drive"],
            "drive":    ["finish"]
        }

    def transition(self, new_state):
        if new_state not in self.transitions[self.state]:
            self.pending = None
            self.count = 0
            return

        if new_state == self.pending:
            self.count += 1   # condition held again, keep counting
        else:
            self.pending = new_state
            self.count = 1    # new candidate, start fresh

        if self.count >= self.debounce[self.state]:
            self.state = new_state   # stable long enough, commit the transition
            self.pending = None
            self.count = 0

    def update(self, knee_angle, hip_angle):
        # hip is the better signal here as knee stays high through finish
        if self.state == "drive" and knee_angle > 155 and hip_angle > 110:
            self.transition("finish")

        # hip drops faster than knee coming out of finish — use it as primary trigger
        elif self.state == "finish" and hip_angle < 120:
            self.transition("recovery")

        # both must be compressed to confirm we at the catch
        elif self.state == "recovery" and knee_angle < 60 and hip_angle < 30:
            self.transition("catch")

        # legs start extending = drive begins
        elif self.state == "catch" and knee_angle > 65:
            self.transition("drive")


stroke_machine = StrokeMachine()   


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
            # mmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
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

            stroke_machine.update(knee_angle_l, hip_angle_l)        #activate the machine :D
            current_phase = stroke_machine.state
            phase_ranges = ANGLE_RANGES[current_phase]

            """print(f"state={stroke_machine.state} | knee={int(knee_angle_l)} hip={int(hip_angle_l)}") #debug line for some angle refining"""



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


            if color_to_quality(knee_color_l) == 'optimal':
                optimal_count += 1
            elif color_to_quality(knee_color_l) == 'acceptable':        #we can assume for the sake of simplicity that if the left knee is at a good angle the right knee is also
                acceptable_count += 1                                   # It's hard to imagine this not being true   
            else:
                bad_count += 1   

            if color_to_quality(ankle_color_l) == 'optimal':
                optimal_count += 1
            elif color_to_quality(ankle_color_l) == 'acceptable':      
                acceptable_count += 1                                   
            else:
                bad_count += 1                   

            if color_to_quality(hip_color_l) == 'optimal':
                optimal_count += 1
            elif color_to_quality(hip_color_l) == 'acceptable':      
                acceptable_count += 1                                   
            else:
                bad_count += 1                          #tis a bit silly but sure look




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
                       cv.FONT_HERSHEY_PLAIN, 1.2, PHASE_COLORS[current_phase], 3)

            # Left side angles
            cv.putText(image, f"Hip: {int(hip_angle_l)}",
                       (int(landmarks[JOINTS['hip_l']].x * w) + 10, 
                        int(landmarks[JOINTS['hip_l']].y * h)), 
                       cv.FONT_HERSHEY_PLAIN, 0.5, hip_color_l, 2)
            
            cv.putText(image, f"Knee: {int(knee_angle_l)}", 
                       (int(landmarks[JOINTS['knee_l']].x * w) + 10, 
                        int(landmarks[JOINTS['knee_l']].y * h)), 
                       cv.FONT_HERSHEY_PLAIN, 0.5, knee_color_l, 2)
            
            cv.putText(image, f"Ankle: {int(ankle_angle_l)}", 
                       (int(landmarks[JOINTS['ankle_l']].x * w) + 10, 
                        int(landmarks[JOINTS['ankle_l']].y * h)), 
                       cv.FONT_HERSHEY_PLAIN, 0.5, ankle_color_l, 2)

            # Right side angles
            cv.putText(image, f"Hip: {int(hip_angle_r)}",
                       (int(landmarks[JOINTS['hip_r']].x * w) - 80, 
                        int(landmarks[JOINTS['hip_r']].y * h)), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, hip_color_r, 2)
            
            cv.putText(image, f"Knee: {int(knee_angle_r)}", 
                       (int(landmarks[JOINTS['knee_r']].x * w) - 80, 
                        int(landmarks[JOINTS['knee_r']].y * h)), 
                       cv.FONT_HERSHEY_PLAIN, 0.5, knee_color_r, 2)
            
            cv.putText(image, f"Ankle: {int(ankle_angle_r)}", 
                       (int(landmarks[JOINTS['ankle_r']].x * w) - 80, 
                        int(landmarks[JOINTS['ankle_r']].y * h)), 
                       cv.FONT_HERSHEY_PLAIN, 0.5, ankle_color_r, 2)

            #put the rowing stroke phase ideal angles at the bottom of screen. 
            guidelines_y = h - 100
            cv.rectangle(image, (10, guidelines_y - 10), (w - 10, h - 10), (0, 0, 0), -1)
            
            cv.putText(image, f"{current_phase.upper()} Guidelines:", 
                       (20, guidelines_y + 10),
                       cv.FONT_HERSHEY_PLAIN, 0.6, (255, 255, 255), 2)
            
            cv.putText(image, f"Knee: {phase_ranges['knee']['optimal'][0]}-{phase_ranges['knee']['optimal'][1]}deg", 
                       (20, guidelines_y + 35),
                       cv.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
            
            cv.putText(image, f"Hip: {phase_ranges['hip']['optimal'][0]}-{phase_ranges['hip']['optimal'][1]}deg", 
                       (200, guidelines_y + 35),
                       cv.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
            
            cv.putText(image, f"Ankle: {phase_ranges['ankle']['optimal'][0]}-{phase_ranges['ankle']['optimal'][1]}deg", 
                       (380, guidelines_y + 35),
                       cv.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)

        cv.imshow('Rowing Pose Overlay', image)

        if cv.waitKey(7) & 0xFF == 27:
            break

cap.release()
cv.destroyAllWindows()

print("============================")
print("===FORM REVIEW===")
print("============================")


if optimal_count > acceptable_count and bad_count: 
    print("Overall your form is optimal")
elif acceptable_count > optimal_count and bad_count:
        print("Overall your form is acceptable")
else: 
    print("Overall your form is bad :(")