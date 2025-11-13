import cv2
import mediapipe as mp
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open webcam")
    exit()
print("Webcam opened successfully")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1
)

# Create a black canvas
ret, frame = cap.read()
if not ret:
    print("Failed to grab frame")
    cap.release()
    exit()
canvas = np.zeros_like(frame)

# Brush color and thickness
color = (0, 0, 255)  # default red
brush_thickness = 2

# Previous coordinates for smooth drawing
prev_right = None

# Resize parameters
webcam_width, webcam_height = 900, 540

# Helper: count fingers
def count_fingers(hand_landmarks):
    fingers = []
    tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers.count(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hand_handedness.classification[0].label  # 'Right' or 'Left'
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            index_x = int(hand_landmarks.landmark[8].x * w)
            index_y = int(hand_landmarks.landmark[8].y * h)
            finger_count = count_fingers(hand_landmarks)

            # ✋ LEFT HAND: color control & clear canvas
            if label == "Left":
                if finger_count == 0:  # Fist = clear canvas
                    canvas = np.zeros_like(frame)
                    print("🧹 Canvas cleared!")
                elif finger_count == 1:
                    color = (0, 0, 255)  # Red
                elif finger_count == 2:
                    color = (0, 255, 0)  # Green
                elif finger_count == 3:
                    color = (255, 0, 0)  # Blue
                elif finger_count == 4:
                    color = (0, 255, 255)  # Yellow
                elif finger_count == 5:
                    color = (255, 255, 255)  # White

            # 🤚 RIGHT HAND: drawing control (only index finger up)
            elif label == "Right":
                # Draw only if index finger up and other fingers down
                if (hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y) and finger_count == 1:
                    if prev_right is None:
                        prev_right = (index_x, index_y)
                    cv2.line(canvas, prev_right, (index_x, index_y), color, brush_thickness)
                    prev_right = (index_x, index_y)
                else:
                    prev_right = None

    # Resize webcam feed for display
    resized_frame = cv2.resize(frame, (webcam_width, webcam_height))

    # Combine frames for display
    cv2.imshow("Webcam Feed", resized_frame)
    cv2.imshow("Air Canvas", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
