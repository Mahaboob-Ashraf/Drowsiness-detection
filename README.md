# Real-Time Drowsiness Detection System

A Python-based system that uses a webcam to monitor a driver's eyes in real-time. It calculates the Eye Aspect Ratio (EAR) to detect signs of drowsiness and triggers a flashing visual and audio alarm to alert the driver, preventing potential accidents.

![Drowsiness Detection Demo](./demo.gif)

---

## Key Features

- **Real-Time Eye Tracking**: Accurately tracks eye landmarks using a standard webcam.
- **EAR-Based Detection**: Implements the Eye Aspect Ratio (EAR) algorithm to reliably detect blinks and prolonged eye closure.
- **Audio-Visual Alerts**: Triggers a loud audio alarm and a flashing red screen overlay with text to alert the user.
- **Responsive System**: The alarm plays continuously while drowsiness is detected and stops instantly when eyes reopen.

---

## Technologies Used

- Python
- OpenCV
- Mediapipe
- NumPy
- Pygame

---

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/drowsiness-detection.git](https://github.com/your-username/drowsiness-detection.git)
    cd drowsiness-detection
    ```

2.  **Install dependencies:**
    ```powershell
    py -m pip install numpy "opencv-python==4.9.0.80" mediapipe pygame
    ```

3.  **Add an alarm sound:**
    - Place an audio file named `alarm.mp3` in the project's root folder.

4.  **Run the script:**
    ```powershell
    py Drowsiness_detector_v1.py
    ```
    - Press `q` with the video window active to quit.
