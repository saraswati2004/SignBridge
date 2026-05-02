from function import *
import cv2
import os
import numpy as np

# Create folders
for action in actions:
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
) as hands:

    for action in actions:
        image_folder = os.path.join("Image", action)

        if not os.path.exists(image_folder):
            print(f"Folder not found: {image_folder}")
            continue

        image_files = sorted(os.listdir(image_folder), key=lambda x: int(os.path.splitext(x)[0]))

        for file in image_files:
            image_path = os.path.join(image_folder, file)
            frame = cv2.imread(image_path)

            if frame is None:
                print(f"Warning: Image not found -> {image_path}")
                continue

            # Detect hand
            image, results = mediapipe_detection(frame, hands)

            if not results.multi_hand_landmarks:
                print(f"No hand detected for {action} image {file}")
                continue # Skip this image
            
            print(f"Hand detected for {action} image {file}")

            # Draw landmarks
            draw_styled_landmarks(image, results)

            # Show image
            message = f"Collecting {action} Image {file}"
            cv2.putText(image, message, (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)

            # Extract keypoints
            keypoints = extract_keypoints(results)

            # Save .npy file using same filename
            file_number = os.path.splitext(file)[0]
            npy_path = os.path.join(DATA_PATH, action, f"{file_number}.npy")
            np.save(npy_path, keypoints)

            # Wait a little so you can see image
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

cv2.destroyAllWindows()