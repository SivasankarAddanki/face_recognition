# Face Recognition System

A simple face-recognition application built with Python, OpenCV, and Tkinter.
It uses the computer's webcam to capture face samples, train an LBPH face
recognizer, and identify trained people in real time.

## Features

- Simple Tkinter interface
- Webcam-based face capture
- Captures 100 samples for each person
- Supports multiple people
- Automatic dataset expansion for older datasets with fewer than 100 images
- Histogram equalization for more stable recognition under different lighting
- LBPH face-recognition model
- Displays `Unknown` when a face does not meet the confidence threshold
- Saves the trained model inside the project

## Technologies

- Python
- OpenCV Contrib
- NumPy
- Tkinter
- Haar Cascade face detection
- LBPH face recognition

## Project Structure

```text
face_recognition/
|-- dataset/
|   |-- person_name/
|   |   |-- 001.jpg
|   |   `-- ...
|-- model/
|   |-- trainer.yml
|   `-- labels.json
|-- haarcascade_frontalface_default.xml
|-- train.py
|-- requirements.txt
|-- run_face_recognition.bat
`-- README.md
```

The `dataset/` and `model/` directories are created or updated by the
application.

## Requirements

- Windows 10 or Windows 11
- Python 3.10 or newer
- A working webcam
- Git

## Installation

Open PowerShell and run:

```powershell
cd "$HOME\Desktop"
git clone https://github.com/SivasankarAddanki/face_recognition.git
cd face_recognition
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Execute The Application

With the virtual environment activated:

```powershell
python train.py
```

You can also execute it without activating the environment:

```powershell
.\.venv\Scripts\python.exe train.py
```

On Windows, you can alternatively double-click:

```text
run_face_recognition.bat
```

## Capture And Train A Face

1. Start the application.
2. Enter the person's name.
3. Click **Capture Face and Train**.
4. Look toward the webcam while 100 samples are captured.
5. Slowly turn the face slightly left and right and vary the expression.
6. Keep the face visible and use steady lighting.
7. Wait for the training-complete message.

The application retains previously captured people and replaces the model with
a newly trained model containing every person in `dataset/`.

## Recognize A Face

1. Start the application.
2. Click **Recognize Face**.
3. Look toward the webcam.
4. The detected name and confidence distance appear above the face.
5. Press `Q` while the camera window is selected to stop recognition.

Lower confidence-distance values indicate a closer match. Faces above the
configured threshold are displayed as `Unknown`.

## Generated Files

Captured samples:

```text
dataset/<person_name>/*.jpg
```

Trained LBPH model:

```text
model/trainer.yml
```

Label mapping:

```text
model/labels.json
```

Training again replaces `model/trainer.yml` and `model/labels.json`.

## Troubleshooting

### Camera does not open

- Close applications that may already be using the webcam.
- Allow camera access in Windows privacy settings.
- Restart the application.

### `cv2.face` is missing

Remove standard OpenCV and install the contribution package:

```powershell
pip uninstall -y opencv-python opencv-contrib-python
pip install opencv-contrib-python
```

### Recognition is inaccurate

- Capture samples under even lighting.
- Keep only one face in view during training.
- Include small head-angle and expression changes.
- Capture the person again and retrain the model.

## Privacy

Face images and the trained model are stored locally. Do not publish biometric
data without permission from every person included in the dataset.
