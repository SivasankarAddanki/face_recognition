# Face Recognition Project Learning Guide

This guide explains the project in a way that helps you understand the code,
demonstrate it, and answer questions about it.

## 1. Project Goal

The application performs three main jobs:

1. Detect a face from the webcam.
2. Capture samples and train a local recognition model.
3. Recognize trained people in real time.

The project runs locally. It does not require a cloud API or internet
connection after the dependencies are installed.

## 2. Technologies Used

| Technology | Purpose |
| --- | --- |
| Python | Main programming language |
| OpenCV | Camera access, face detection, image processing, and recognition |
| OpenCV Contrib | Provides the LBPH face recognizer through `cv2.face` |
| NumPy | Stores labels in the numeric format required during training |
| Tkinter | Creates the desktop user interface |
| Haar Cascade | Finds face locations inside camera frames |
| LBPH | Learns local facial texture patterns and predicts identities |

## 3. Important Project Files

| Path | Description |
| --- | --- |
| `train.py` | Main application, capture, training, and recognition code |
| `haarcascade_frontalface_default.xml` | Pretrained frontal-face detector |
| `dataset/<name>/` | Face samples organized by person |
| `model/trainer.yml` | Saved LBPH recognition model |
| `model/labels.json` | Maps numeric model labels to person names |
| `requirements.txt` | Python dependencies |
| `run_face_recognition.bat` | Windows launcher |

## 4. Detection Versus Recognition

Face detection and face recognition are different:

- **Detection** answers: "Where is a face in this frame?"
- **Recognition** answers: "Whose detected face is this?"

The Haar Cascade performs detection. LBPH performs recognition.

## 5. End-To-End Workflow

```text
Webcam frame
    |
Convert frame to grayscale
    |
Haar Cascade detects a face
    |
Crop and resize the detected face to 200 x 200
    |
Normalize lighting with histogram equalization
    |
Train LBPH or send the face to the saved LBPH model
    |
Return a numeric label and confidence distance
    |
Map the label to a name using labels.json
```

## 6. Code Walkthrough

### Project Paths

The application builds paths relative to `train.py`:

```python
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "model"
```

This allows the application to run even when the project is moved to another
folder.

### `preprocess_face`

This function:

1. Resizes every face to `200 x 200`.
2. Applies histogram equalization.

Equal image sizes are required for consistent training. Histogram equalization
reduces some effects caused by bright and dark lighting.

### `expand_person_dataset`

Older datasets may contain fewer than 100 samples. This function creates mild
variations using:

- Small clockwise and anticlockwise rotations
- Small horizontal shifts
- Small vertical shifts

Real camera samples are preferable. Augmentation is only a fallback for a
smaller existing dataset.

### `create_recognizer`

This function checks that `cv2.face` exists and creates an LBPH recognizer:

```python
cv2.face.LBPHFaceRecognizer_create()
```

The `opencv-contrib-python` package is required because standard
`opencv-python` may not provide this module.

### `open_camera`

The application first tries the Windows DirectShow backend:

```python
cv2.VideoCapture(0, cv2.CAP_DSHOW)
```

If that fails, it retries with the default OpenCV camera backend.

### `load_detector`

This loads the Haar Cascade XML file. The detector searches each grayscale
frame for frontal faces.

### `capture_face`

The capture process:

1. Creates `dataset/<person_name>/`.
2. Removes old samples for the same person.
3. Opens the webcam.
4. Detects the largest visible face.
5. Crops and resizes that face.
6. Saves 100 JPEG samples.

Pressing `Q` cancels the camera loop. At least 10 samples are required before
the partial capture can be accepted.

### `train_model`

Training works as follows:

1. Read each non-empty person folder.
2. Assign a numeric label to each person.
3. Preprocess every image.
4. Train the LBPH model.
5. Save the model to `model/trainer.yml`.
6. Save the name mapping to `model/labels.json`.

Running training again replaces the previous model.

### `recognize_faces`

During recognition, the application:

1. Loads `trainer.yml` and `labels.json`.
2. Detects faces in each webcam frame.
3. Preprocesses each face.
4. Calls `recognizer.predict(face)`.
5. Displays the mapped name when the distance is within the threshold.
6. Displays `Unknown` for a weak match.

Press `Q` to stop recognition.

### `FaceRecognitionApp`

This Tkinter class creates:

- A name field
- A **Capture Face and Train** button
- A **Recognize Face** button
- A status message

The buttons connect the user interface to the capture, training, and
recognition functions.

## 7. Understanding LBPH

LBPH means **Local Binary Patterns Histograms**.

In simplified terms:

1. The image is divided into local regions.
2. Each pixel is compared with nearby pixels.
3. These comparisons produce binary texture patterns.
4. Histograms summarize the patterns in each region.
5. Recognition compares the new face histogram with trained histograms.

LBPH is useful for a small offline project because it is:

- Fast on a normal CPU
- Easy to train
- Suitable for grayscale images
- Available directly in OpenCV Contrib

It is less reliable than modern deep-learning face embeddings when faces have
large pose, age, lighting, or camera-quality differences.

## 8. Confidence Distance

OpenCV LBPH returns a distance, not a percentage.

- A lower value means a closer match.
- A higher value means a weaker match.
- The project accepts a match at or below `65.0`.
- Values above the threshold are shown as `Unknown`.

The threshold should be tested using images that were not used for training.

## 9. How To Run

```powershell
git clone https://github.com/SivasankarAddanki/face_recognition.git
cd face_recognition
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
```

Without activating the virtual environment:

```powershell
.\.venv\Scripts\python.exe train.py
```

## 10. How To Demonstrate The Project

1. Explain the difference between detection and recognition.
2. Start the program and enter a new name.
3. Capture samples while slightly changing head angle and expression.
4. Explain that training replaces the model using all non-empty datasets.
5. Run recognition for a trained person.
6. Test an untrained person and discuss the `Unknown` threshold.
7. Show `dataset/`, `trainer.yml`, and `labels.json`.

## 11. Testing Checklist

- The webcam opens.
- A face rectangle appears.
- Exactly 100 images are captured for a new person.
- The model and labels files are created.
- Every trained person is recognized.
- An untrained person is displayed as `Unknown`.
- Pressing `Q` closes each camera window.
- Retraining preserves other non-empty person folders.

Do not calculate accuracy using the same images used for training. Use
separate test images or a held-out portion of the captured dataset.

## 12. Current Repository Data Note

The saved `model/trainer.yml` and `model/labels.json` contain these labels:

- `praveen`
- `shiva`
- `vinay`

At the time this guide was written, only Praveen's 100 source images remained
in the repository dataset. Shiva and Vinay remain encoded in the saved model,
but their original source images are not available in the repository.
Retraining from the current dataset will therefore create a new model using
only non-empty dataset folders.

## 13. Limitations

- Haar Cascade works best with frontal faces.
- LBPH can be sensitive to major pose and appearance changes.
- Consecutive webcam frames may be very similar.
- The application has no liveness or anti-spoofing check.
- A printed photo may be accepted as a face.
- The model is intended for a small local dataset, not large-scale identity
  systems.
- Recognition results should not be used for security-critical decisions.

## 14. Possible Improvements

- Capture images at timed intervals to increase variation.
- Add blur and brightness quality checks before saving samples.
- Use separate training and testing datasets.
- Add model evaluation with a confusion matrix.
- Replace Haar Cascade with a modern DNN face detector.
- Replace LBPH with face embeddings and distance comparison.
- Add liveness detection.
- Add a person-management screen for deleting and retraining identities.
- Move camera processing to a worker thread so the GUI remains responsive.

## 15. Common Questions

### Why convert images to grayscale?

Haar Cascade and LBPH can work effectively with intensity patterns. Removing
color also reduces data and processing cost.

### Why resize every face?

The model needs a consistent input shape. Different face crop sizes would make
the extracted patterns less comparable.

### Why use histogram equalization?

It improves contrast and reduces some differences caused by lighting.

### Why save labels separately?

The model predicts numeric IDs. `labels.json` converts those IDs into readable
names.

### What happens when a new person is trained?

The new samples are stored in a new folder. The application then trains a
replacement model from all non-empty person folders.

### Is 100 training images the same as 100 independent examples?

Not necessarily. Webcam frames captured close together can be very similar.
Quality and variation matter more than simply increasing the count.

## 16. Privacy And Responsible Use

Face images and trained models are biometric data. Obtain clear permission
before collecting or publishing another person's face. Restrict access to the
dataset and model, and delete them when they are no longer needed.
