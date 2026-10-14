# Driver Drowsiness Detection using OpenCV and MediaPipe

**CSC 126 – Computer Vision | Midterm Project**
Caraga State University – College of Computing and Information Sciences

## Team

| Member | Topic / Contribution |
|---|---|
| Eddie Rufo II Gause | f. Facial Landmark Detection (eye landmarks, EAR, drowsiness logic) |
| Geuel Morillon | c. Haar Cascade Classifier (face detection) |
| John Pintasan | a. Image Operations (preprocessing) |

## Project Description

This application watches a live webcam feed and alerts the user when their eyes stay closed for too long, a common sign of drowsiness. It is meant for a simple real-world use case: warning a driver or a student studying late at night before they fall asleep.

**Objectives**
- Detect a face in real time from a webcam feed.
- Locate the eyes and measure how open they are.
- Trigger a clear visual alert when the eyes stay closed longer than a set time.
- Show the processed frames and results (boxes, points, labels, overlays) on screen.

## Techniques Used (Laboratory Topics)

| Topic | What the project does with it |
|---|---|
| **a. Image Operations** | Converts frames to grayscale, applies a Gaussian blur to reduce noise, applies CLAHE (contrast-limited adaptive histogram equalization) to even out lighting, crops and resizes the face region, flips the frame into a mirror view, and draws the boxes, text and overlays. |
| **c. Haar Cascade Classifier** | `haarcascade_frontalface_default.xml` finds the face. The detected face region is cropped and passed to the landmark step. |
| **f. Facial Landmark Detection** | MediaPipe Face Landmarker finds the face landmarks. Six points per eye are used to compute the Eye Aspect Ratio (EAR). |

### How it works

1. **Capture:** a frame is read from the webcam and mirrored.
2. **Preprocess (a):** the frame is converted to grayscale, blurred, and contrast-equalized with CLAHE.
3. **Detect the face (c):** the Haar cascade finds the largest face. If it misses for a few frames, the last known box is kept briefly so the display doesn't flicker.
4. **Landmarks (f):** the face region is cropped and resized, then MediaPipe finds the facial landmarks. Eye points are mapped back to full-frame coordinates.
5. **Eye Aspect Ratio:** for each eye,

   ```
   EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
   ```

   The two eyes are averaged. An open eye gives a higher EAR, and a closed eye gives a value near 0.
6. **Decision:** if the EAR stays below `EAR_THRESHOLD` for `CLOSED_SECONDS`, the status becomes **DROWSY! WAKE UP!** and a red border appears. Normal blinks are too short to trigger it.
7. **Display and log:** the face box, eye points, EAR value, status and a small preprocessed-image preview are drawn on the frame. Each drowsy episode is logged to `drowsiness_log.csv`.

### Status messages

| Status | Meaning |
|---|---|
| `Awake` (green) | Eyes open |
| `Eyes closed` (orange) | Eyes closed, but not long enough to trigger the alert |
| `DROWSY! WAKE UP!` (red + red border) | Eyes closed for at least `CLOSED_SECONDS` |
| `NO FACE` (gray) | No face found |

## Requirements

- Python 3 (developed on Windows with Python 3.14)
- A working webcam
- The packages in `requirements.txt`:
  - `opencv-contrib-python` (version 4.x, see the note below)
  - `mediapipe`

> **Important:** install OpenCV **4.x**, not OpenCV 5. In OpenCV 5 the Haar cascade classifier was moved out of the main module, and the program stops with `Could not load Haar cascade`. `requirements.txt` already pins `opencv-contrib-python<5`.

## Setup

1. **Clone the repository**
   ```
   git clone https://github.com/cnomadx/drowsiness-detector.git
   cd drowsiness-detector
   ```

2. **Install the dependencies**
   ```
   pip install -r requirements.txt
   ```
   If you previously installed `opencv-python` or OpenCV 5, remove them first so they don't conflict:
   ```
   pip uninstall -y opencv-python opencv-contrib-python
   pip install -r requirements.txt
   ```

3. **Check that the model file is present.** `face_landmarker.task` must be in the same folder as `5pipeline.py`. If it is missing, download it:
   ```
   curl -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
   ```

4. **(Optional) verify the Haar cascade loads**
   ```
   python -c "import cv2; print(cv2.__version__); c = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'); print('cascade loaded:', not c.empty())"
   ```
   You should see a version starting with `4.` and `cascade loaded: True`.

## How to Run

Run the program **from the project folder**, so it can find `face_landmarker.task`:

```
python 5pipeline.py
```

A window opens with your webcam feed.

| Key | Action |
|---|---|
| `q` | Quit |
| `s` | Save a screenshot to the `screenshots/` folder |

Tips: sit facing the camera in even lighting, and keep your whole face in view.

## Settings

These values are at the top of `5pipeline.py` and can be tuned:

| Setting | Default | Meaning |
|---|---|---|
| `EAR_THRESHOLD` | `0.20` | EAR below this counts as eyes closed |
| `CLOSED_SECONDS` | `1.5` | Seconds of closed eyes before the drowsy alert |
| `FACE_MARGIN` | `0.25` | Extra space around the Haar face box before landmarking |
| `CROP_SIZE` | `256` | Size the face crop is resized to before landmarking |
| `LOST_GRACE_FRAMES` | `5` | Frames to keep the last face box when the Haar cascade misses |

The default threshold was chosen from a measurement on one team member: EAR was about 0.31 with eyes open and about 0.02 with eyes closed. Other people may need a different value.

## Project Structure

```
.
├── 5pipeline.py          # Final application (topics a, c, f)
├── face_landmarker.task  # MediaPipe pre-trained landmark model
├── requirements.txt      # Python dependencies
├── README.md
├── 1webcam.py            # Development step 1: webcam test
├── 2landmarks.py         # Development step 2: draw facial landmarks
├── 3ear.py               # Development step 3: eye aspect ratio
├── 4drowsiness.py        # Development step 4: landmark-only drowsiness detection
├── drowsiness_log.csv    # Log of drowsy events (created when the app runs)
└── screenshots/          # Output screenshots (saved with the `s` key)
```

Files `1webcam.py` to `4drowsiness.py` are the earlier stages of the project and can be run on their own. `5pipeline.py` combines everything.

## Output

The application window shows:
- the processed frame, with a bounding box around the detected face
- the eye landmark points, the live EAR value and a status label
- a red border when the drowsy alert is active
- a small grayscale/CLAHE preview in the bottom-right corner (the image operations stage)

Screenshots of the output are in the `screenshots/` folder.

## Known Limitations

- The Haar cascade works best with a mostly frontal face. Strongly turned or tilted heads can be missed.
- Faces smaller than 100 x 100 pixels are ignored, so the user must be reasonably close to the camera.
- Very dim or uneven lighting, glasses glare, and sunglasses can affect detection and the EAR values.
- The EAR threshold was calibrated on one person and may need adjusting for others.
- Only one face (the largest) is analysed at a time.
- Looking down (for example at a phone) can lower the measured EAR and may look like closed eyes.

## Possible Improvements

- Per-user automatic calibration of the EAR threshold.
- A head-pose check to separate looking down from closing the eyes.
- An audio alarm in addition to the visual alert.
- Detecting yawning with mouth landmarks.

## Credits and References

- [OpenCV](https://opencv.org/): image processing and Haar cascade classifier
- [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker): pre-trained facial landmark model (used as provided, not trained by the team)
- Eye Aspect Ratio method: Soukupová and Čech, "Real-Time Eye Blink Detection using Facial Landmarks" (2016)
