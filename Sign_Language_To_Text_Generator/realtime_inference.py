# realtime_inference.py
# Usage: python realtime_inference.py --model asl_model.h5 --encoder label_encoder.joblib

import cv2
import numpy as np
import mediapipe as mp
import argparse
import joblib
import time
from collections import deque, Counter
import tensorflow as tf

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_landmarks_from_frame(frame, hands):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if not result.multi_hand_landmarks:
        return None
    lm = result.multi_hand_landmarks[0].landmark
    coords = []
    for l in lm:
        coords += [l.x, l.y, l.z]
    return np.array(coords, dtype=np.float32)

def main(model_path, encoder_path, smoothing=6):
    model = tf.keras.models.load_model(model_path)
    le = joblib.load(encoder_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    sentence = []
    frame_buffer = deque(maxlen=smoothing)
    last_save_time = 0
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                        min_detection_confidence=0.55, min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            lm = get_landmarks_from_frame(frame, hands)
            if lm is not None:
                pred = model.predict(lm.reshape(1, -1), verbose=0)
                idx = np.argmax(pred[0])
                label = le.inverse_transform([idx])[0]
                frame_buffer.append(label)
                # draw landmarks
                # re-run hands to get landmarks for drawing (we did above but not stored)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(display_frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                # smoothing: pick most common in buffer
                common = Counter(frame_buffer).most_common(1)[0][0]
                # simple logic: add to sentence if stable and not duplicate last
                if len(sentence) == 0 or common != sentence[-1]:
                    # require buffer full for stability
                    if len(frame_buffer) == frame_buffer.maxlen:
                        sentence.append(common)
            else:
                # when no hand, optionally insert space or ignore
                # here we ignore; user can press key to add space
                pass

            # UI overlay: show current predictions and sentence
            cv2.rectangle(display_frame, (0,0), (640,60), (0,0,0), -1)
            text = " ".join(sentence[-20:])  # show last 20 tokens
            cv2.putText(display_frame, f"Detected: {frame_buffer[-1] if frame_buffer else '-'}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(display_frame, f"Sentence: {text}", (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            cv2.imshow("ASL -> Text", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # clear sentence
                sentence = []
            elif key == ord(' '):
                # add space (useful for separating words if you map multiple-letter gestures to form a word)
                sentence.append(" ")
            elif key == ord('s'):
                # save text to file
                now = int(time.time())
                fname = f"recognized_{now}.txt"
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(" ".join(sentence))
                print("Saved to", fname)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--encoder", type=str, required=True)
    parser.add_argument("--smoothing", type=int, default=6)
    args = parser.parse_args()
    main(args.model, args.encoder, smoothing=args.smoothing)
