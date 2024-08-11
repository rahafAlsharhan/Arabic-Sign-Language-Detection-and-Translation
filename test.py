import cv2
import mediapipe as mp
import pyautogui
import webbrowser
import time
import tkinter as tk


def toggle_eye_tracking():
    global eye_tracking_enabled

    if eye_tracking_enabled:
        eye_tracking_enabled = False
        toggle_button.config(text="Enable Eye Tracking")
        webbrowser.open("img.html", new=0, autoraise=True)
    else:
        eye_tracking_enabled = True
        toggle_button.config(text="Disable Eye Tracking")
        webbrowser.open("img.html", new=0, autoraise=True)


def process_frame():
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    if landmark_points and eye_tracking_enabled:
        landmarks = landmark_points[0].landmark
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))
            if id == 1:
                screen_x = screen_w * landmark.x
                screen_y = screen_h * landmark.y
                pyautogui.moveTo(screen_x, screen_y)
        left = [landmarks[145], landmarks[159]]
        for landmark in left:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))
        if (left[0].y - left[1].y) < 0.004:
            pyautogui.click()
            pyautogui.sleep(1)

    cv2.imshow('Eye Controlled Mouse', frame)
    cv2.waitKey(1)


# Initialize eye tracking toggle state
eye_tracking_enabled = False

# Open web page in a new tab
webbrowser.open_new_tab("index.html")

# Initialize video capture and mediapipe
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

# Create GUI window
window = tk.Tk()
window.title("Eye Tracking Control")

# Create toggle button
toggle_button = tk.Button(window, text="Enable Eye Tracking", command=toggle_eye_tracking)
toggle_button.pack(pady=10)

# Start processing frames
while True:
    process_frame()
    window.update()

    # Break the loop if the window is closed
    if cv2.waitKey(1) == ord('q') or not window.winfo_exists():
        break

# Release resources
cam.release()
cv2.destroyAllWindows()