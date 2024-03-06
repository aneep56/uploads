import cv2
import mediapipe as mp
import time
import pyautogui

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define the keyboard layout
keyboard = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]
]

# Function to draw keyboard
def draw_keyboard(frame):
    key_width = 40
    key_height = 60
    key_padding = 10
    start_x = 70
    start_y = 70

    for i, row in enumerate(keyboard):
        for j, key in enumerate(row):
            x = start_x + j * (key_width + key_padding)
            y = start_y + i * (key_height + key_padding)
            cv2.rectangle(frame, (x, y), (x + key_width, y + key_height), (255, 255, 255), 2)
            cv2.putText(frame, key, (x + 10, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Function to detect key press
def detect_key_press(hand_landmarks, frame):
    for i, row in enumerate(keyboard):
        for j, key in enumerate(row):
            x = 70 + j * (40 + 10)
            y = 70 + i * (60 + 10)
            if (hand_landmarks.landmark[8].x * frame.shape[1] > x and
                    hand_landmarks.landmark[8].x * frame.shape[1] < x + 40 and
                    hand_landmarks.landmark[8].y * frame.shape[0] > y and
                    hand_landmarks.landmark[8].y * frame.shape[0] < y + 60 and
                    hand_landmarks.landmark[12].x * frame.shape[1] > x and
                    hand_landmarks.landmark[12].x * frame.shape[1] < x + 40 and
                    hand_landmarks.landmark[12].y * frame.shape[0] > y and
                    hand_landmarks.landmark[12].y * frame.shape[0] < y + 60):
                return key
    return None

# Main loop
cap = cv2.VideoCapture(0)

# Increase window size by 30%
ret, frame = cap.read()
frame_height, frame_width, _ = frame.shape
window_width = int(frame_width * 1.3)
window_height = int(frame_height * 1.3)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Frame', window_width, window_height)

# Initialize text box content
text_box_content = ""

# Initialize time for tracking duration of index finger hovering
start_hover_time = 0
hover_duration_threshold = 2  # Threshold time for hovering (in seconds)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe
    results = hands.process(rgb_frame)

    # Draw the keyboard on the frame
    draw_keyboard(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detect key press
            key_pressed = detect_key_press(hand_landmarks, frame)
            if key_pressed:
                text_box_content += key_pressed
                print(f"Key Pressed: {key_pressed}")
                start_hover_time = time.time()  # Start tracking hover time

            # If index finger hovers over a key for a certain duration, type the character
            if start_hover_time != 0 and time.time() - start_hover_time > hover_duration_threshold:
                pyautogui.press(key_pressed.lower())  # Simulate key press
                start_hover_time = 0  # Reset hover time

    # Display frame
    cv2.imshow('Frame', frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
