import os
import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

### CONSTANT - CHANGE IF NECESSARY
SEQUENCE_LENGTH = 20
IMG_HEIGHT, IMG_WIDTH = 64, 64
CLASSES_LIST = ["WalkingWithDog", "PlayingGuitar", "Swing", "HorseRiding"]
model = load_model('convlstm_model.h5')
###
def play_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Processed Video', frame)
        
        # Press 'q' to exit the video playback
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def predict_on_video(video_path, output_path, SEQUENCE_LENGTH):
    video_reader = cv2.VideoCapture(video_path)

    default_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    default_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    default_video_fps = int(video_reader.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, default_video_fps, (default_video_width, default_video_height))

    frame_queue = deque(maxlen=SEQUENCE_LENGTH)

    predicted_class = ''

    while video_reader.isOpened():
        ret, frame = video_reader.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

        normalized_frame = resized_frame / 255.0

        frame_queue.append(normalized_frame)

        if len(frame_queue) == SEQUENCE_LENGTH:
            predicted_probs = model.predict(np.expand_dims(frame_queue, axis=0))[0]
            predicted_id = np.argmax(predicted_probs)
            predicted_class = CLASSES_LIST[predicted_id]

        frame_with_prediction = frame.copy()

        cv2.putText(frame_with_prediction, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame_with_prediction)
    video_reader.release()
    out.release()

def real_time_action_recognition():
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    frame_queue = deque(maxlen=SEQUENCE_LENGTH)
    predicted_class = "Waiting..."

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        resized_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        normalized_frame = resized_frame / 255.0

        frame_queue.append(normalized_frame)

        if len(frame_queue) == SEQUENCE_LENGTH:
            predicted_probs = model.predict(np.expand_dims(frame_queue, axis=0))[0]
            predicted_id = np.argmax(predicted_probs)
            predicted_class = CLASSES_LIST[predicted_id]

        cv2.putText(frame, f"Action: {predicted_class}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Real-Time Action Recognition", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    output_video_path = 'video_result/output.mp4'
    predict_on_video('video_test/play_guitar_test.mp4', f'{output_video_path}', SEQUENCE_LENGTH)
    play_video(output_video_path)
    # real_time_action_recognition()