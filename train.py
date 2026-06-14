"""Minimal local face capture, training, and recognition application."""

from __future__ import annotations

import json
import re
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "trainer.yml"
LABELS_PATH = MODEL_DIR / "labels.json"
CASCADE_PATH = ROOT / "haarcascade_frontalface_default.xml"

SAMPLE_COUNT = 100
RECOGNITION_THRESHOLD = 65.0


def preprocess_face(image: np.ndarray) -> np.ndarray:
    """Normalize face size and lighting for more stable LBPH matching."""
    resized = cv2.resize(image, (200, 200))
    return cv2.equalizeHist(resized)


def expand_person_dataset(person_dir: Path, target_count: int = SAMPLE_COUNT) -> int:
    """Create mild pose variations when an existing dataset has fewer images."""
    source_paths = sorted(person_dir.glob("*.jpg"))
    if not source_paths:
        return 0

    current_count = len(source_paths)
    variations = (
        (-6.0, -3.0, 0.0),
        (6.0, 3.0, 0.0),
        (-3.0, 2.0, -2.0),
        (3.0, -2.0, 2.0),
    )

    while current_count < target_count:
        source_path = source_paths[current_count % len(source_paths)]
        image = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            current_count += 1
            continue

        angle, shift_x, shift_y = variations[current_count % len(variations)]
        height, width = image.shape
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        matrix[0, 2] += shift_x
        matrix[1, 2] += shift_y
        augmented = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        current_count += 1
        cv2.imwrite(str(person_dir / f"{current_count:03d}.jpg"), augmented)

    return current_count


def create_recognizer():
    if not hasattr(cv2, "face"):
        raise RuntimeError(
            "OpenCV face module is missing. Install opencv-contrib-python."
        )
    return cv2.face.LBPHFaceRecognizer_create()


def open_camera():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open the camera.")
    return camera


def load_detector():
    detector = cv2.CascadeClassifier(str(CASCADE_PATH))
    if detector.empty():
        raise RuntimeError(f"Could not load face detector: {CASCADE_PATH}")
    return detector


def safe_folder_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip()).strip("_")
    if not value:
        raise ValueError("Enter a name using letters or numbers.")
    return value


def capture_face(name: str) -> int:
    person_name = name.strip()
    folder_name = safe_folder_name(person_name)
    person_dir = DATASET_DIR / folder_name
    person_dir.mkdir(parents=True, exist_ok=True)

    for old_image in person_dir.glob("*.jpg"):
        old_image.unlink()

    detector = load_detector()
    camera = open_camera()
    saved = 0

    try:
        while saved < SAMPLE_COUNT:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The camera stopped returning frames.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100)
            )

            if len(faces):
                x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
                face = cv2.resize(gray[y : y + h, x : x + w], (200, 200))
                saved += 1
                cv2.imwrite(str(person_dir / f"{saved:03d}.jpg"), face)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)

            cv2.putText(
                frame,
                f"Samples: {saved}/{SAMPLE_COUNT}  (Q to cancel)",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Capture Face", frame)

            if cv2.waitKey(80) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    if saved < 10:
        raise RuntimeError("Not enough face samples were captured. Please try again.")
    return saved


def train_model() -> tuple[int, int]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    label_names: dict[int, str] = {}

    if not DATASET_DIR.exists():
        raise RuntimeError("No face data found. Capture a face first.")

    people = sorted(
        path
        for path in DATASET_DIR.iterdir()
        if path.is_dir() and any(path.glob("*.jpg"))
    )
    for label, person_dir in enumerate(people):
        expand_person_dataset(person_dir)
        label_names[label] = person_dir.name.replace("_", " ")
        for image_path in sorted(person_dir.glob("*.jpg")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(preprocess_face(image))
                labels.append(label)

    if not images:
        raise RuntimeError("No usable training images were found.")

    recognizer = create_recognizer()
    recognizer.train(images, np.asarray(labels, dtype=np.int32))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    recognizer.write(str(MODEL_PATH))
    LABELS_PATH.write_text(json.dumps(label_names, indent=2), encoding="utf-8")
    return len(images), len(label_names)


def recognize_faces() -> None:
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise RuntimeError("No trained model found. Capture and train a face first.")

    recognizer = create_recognizer()
    recognizer.read(str(MODEL_PATH))
    labels = {
        int(key): value
        for key, value in json.loads(LABELS_PATH.read_text(encoding="utf-8")).items()
    }
    detector = load_detector()
    camera = open_camera()

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The camera stopped returning frames.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100)
            )

            for x, y, w, h in faces:
                face = preprocess_face(gray[y : y + h, x : x + w])
                label, confidence = recognizer.predict(face)
                known = confidence <= RECOGNITION_THRESHOLD and label in labels
                name = labels[label] if known else "Unknown"
                color = (0, 200, 0) if known else (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame,
                    f"{name} ({confidence:.0f})",
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    color,
                    2,
                )

            cv2.putText(
                frame,
                "Press Q to quit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


class FaceRecognitionApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Simple Face Recognition")
        self.root.geometry("460x290")
        self.root.resizable(False, False)

        tk.Label(
            self.root, text="Simple Face Recognition", font=("Segoe UI", 20, "bold")
        ).pack(pady=(24, 18))

        form = tk.Frame(self.root)
        form.pack()
        tk.Label(form, text="Name:", font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=8)
        self.name_entry = tk.Entry(form, width=25, font=("Segoe UI", 11))
        self.name_entry.insert(0, "shiva")
        self.name_entry.pack(side=tk.LEFT)

        tk.Button(
            self.root,
            text="Capture Face and Train",
            command=self.capture_and_train,
            width=28,
            height=2,
        ).pack(pady=(22, 10))
        tk.Button(
            self.root,
            text="Recognize Face",
            command=self.recognize,
            width=28,
            height=2,
        ).pack()

        self.status = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status, fg="#333333").pack(pady=15)

    def capture_and_train(self) -> None:
        try:
            self.status.set("Look at the camera while samples are captured...")
            self.root.update_idletasks()
            captured = capture_face(self.name_entry.get())
            image_count, person_count = train_model()
            self.status.set(f"Model saved: {person_count} person, {image_count} images")
            messagebox.showinfo(
                "Training complete",
                f"Captured {captured} samples and replaced the trained model.",
            )
        except (RuntimeError, ValueError) as error:
            self.status.set("Operation failed")
            messagebox.showerror("Face recognition", str(error))

    def recognize(self) -> None:
        try:
            self.status.set("Recognition running. Press Q in the camera window to stop.")
            self.root.update_idletasks()
            recognize_faces()
            self.status.set("Ready")
        except RuntimeError as error:
            self.status.set("Operation failed")
            messagebox.showerror("Face recognition", str(error))

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    FaceRecognitionApp().run()
