from function import *
from keras.models import model_from_json
import numpy as np
import cv2

# Load model
with open("model_new.json", "r") as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)
model.load_weights("model_new.h5")
print("Model loaded successfully.")

# Colors for probability bars
colors = [(245, 117, 16)] * len(actions)

# Visualization function
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()

    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (20, 80 + num * 35),
                      (20 + int(prob * 300), 105 + num * 35), colors[num], -1)
        cv2.putText(output_frame, f"{actions[num]}: {prob:.2f}",
                    (30, 100 + num * 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame


# Variables
predictions = []
sentence = []
threshold = 0.8 # Increased threshold for higher confidence

# Webcam
cap = cv2.VideoCapture(0)

# Set bigger webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Make window resizable
cv2.namedWindow("OpenCV Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("OpenCV Feed", 1200, 700)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror view (optional but better)
        frame = cv2.flip(frame, 1)

        # ROI (hand area) - MUST MATCH collectdata.py
        x1, y1 = 750, 100
        x2, y2 = 1200, 600

        cropframe = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Detect hand
        image, results = mediapipe_detection(cropframe, hands)

        # Draw landmarks
        draw_styled_landmarks(cropframe, results)

        # Extract keypoints
        keypoints = extract_keypoints(results)

        try:
            # Dense model expects shape (1, 63)
            res = model.predict(np.expand_dims(keypoints, axis=0), verbose=0)[0]
            predictions.append(np.argmax(res))
            
            # Debouncing logic
            # Only if the last 10 predictions are the same and confidence is high
            if np.unique(predictions[-10:]).shape[0] == 1:
                if res[np.argmax(res)] > threshold:
                    predicted_label = actions[np.argmax(res)]
                    
                    # Add to sentence if it's a new letter
                    if len(sentence) == 0 or predicted_label != sentence[-1]:
                        sentence.append(predicted_label)

            # Keep sentence to a reasonable length for display
            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Visualize probabilities
            frame = prob_viz(res, actions, frame, colors)

        except Exception as e:
            print("Prediction error:", e)

        # Output text box
        cv2.rectangle(frame, (0, 0), (1280, 60), (245, 117, 16), -1)
        output_text = f"Output: {' '.join(sentence)}"
        cv2.putText(frame, output_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

        # Show frame
        cv2.imshow("OpenCV Feed", frame)

        # Quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()