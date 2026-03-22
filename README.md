# Rowing Form Analyser

Analyses rowing ergometer footage using MediaPipe pose estimation. Tracks hip, knee, and ankle angles across each phase of the stroke and overlays colour-coded feedback in real time.

**Colours:** green = optimal, yellow = acceptable, red = out of range

---

## Setup

```bash
pip install opencv-python mediapipe numpy
```

---

## Usage

Update the video path in the script, then run:

```python
cap = cv.VideoCapture(r"path/to/your/rowing_footage.mov")
```

```bash
python rowing_analysis.py
```

Press `Esc` to exit. A form summary (optimal / acceptable / bad) is printed to the console on exit. Works best with side-on footage where the full body is visible.

---

## Phase Detection

Phases cycle in order: `finish → recovery → catch → drive → finish → ...`

Each transition requires several consecutive frames to confirm, filtering out noise.

| From | To | Trigger |
|---|---|---|
| drive | finish | knee > 155° and hip > 110° |
| finish | recovery | hip < 120° |
| recovery | catch | knee < 60° and hip < 30° |
| catch | drive | knee > 65° |

---

## Angle Ranges

| Phase | Joint | Optimal | Acceptable |
|---|---|---|---|
| **Catch** | Knee | 35–45° | 30–55° |
| | Hip | 35–50° | 30–60° |
| | Ankle | 75–100° | 70–110° |
| **Drive** | Knee | 60–120° | 50–130° |
| | Hip | 60–120° | 50–130° |
| | Ankle | 85–105° | 75–115° |
| **Finish** | Knee | 163–175° | 150–180° |
| | Hip | 120–145° | 110–150° |
| | Ankle | 85–105° | 75–115° |
| **Recovery** | Knee | 120–160° | 80–170° |
| | Hip | 100–150° | 80–160° |
| | Ankle | 85–105° | 75–115° |
