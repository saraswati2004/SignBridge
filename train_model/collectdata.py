import os
import cv2

directory = r"E:\project\training model\Image"

ACTIONS = ['hello', 'love you','livelong', 'good', 'bad' , 'yes', 'ok', 'peace', 'good luck', 'rockNroll','right', 'left']
ACTION_KEYS = {'1': 'hello', '2': 'love you', '3': 'livelong', '4': 'good',
               '5': 'bad', '6': 'yes', '7': 'ok', '8': 'peace', '9': 'good luck','0': 'rockNroll', 'r': 'right', 'l': 'left'}

for letter in ACTIONS:
    os.makedirs(os.path.join(directory, letter), exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Camera not opened!")
    exit()

print("Press keys 1-0 to capture images")
print("Press ESC to exit")

cv2.namedWindow("Full Frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Full Frame", 1000, 700)
cv2.resizeWindow("ROI", 400, 500)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        continue

    frame = cv2.flip(frame, 1)

    x1, y1 = 750, 100
    x2, y2 = 1200, 600

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)

    roi = frame[y1:y2, x1:x2]

    # Bug 1 fixed: ACTIONS (uppercase) used consistently
    counts = {letter: len(os.listdir(os.path.join(directory, letter))) for letter in ACTIONS}

    cv2.rectangle(frame, (0, 0), (1280, 70), (0, 0, 0), -1)
    cv2.putText(frame, "Press 1-0 to save image | ESC to exit",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

    # Bug 1 fixed: ACTIONS (uppercase) used consistently
    count_text = " | ".join([f"{letter}:{counts[letter]}" for letter in ACTIONS])
    cv2.putText(frame, count_text,
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Full Frame", frame)
    cv2.imshow("ROI", roi)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    # Bug 2 fixed: use ACTION_KEYS mapping with number keys instead of letter matching
    key_char = chr(key)
    if key_char in ACTION_KEYS:
        action = ACTION_KEYS[key_char]
        count = counts[action]
        file_path = os.path.join(directory, action, f"{count}.png")
        cv2.imwrite(file_path, roi)
        print(f"Captured {action}: {count}")

cap.release()
cv2.destroyAllWindows()