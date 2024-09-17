import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
import utils


DATA_PATH = utils.DATA_PATH
signs = utils.signs

mp_holistic = utils.mp_holistic
mp_drawing = utils.mp_drawing


# colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
colors = [(47, 75, 124), (160, 81, 149), (249, 93, 106), (255, 166, 0), (0, 63, 92), (102, 81, 145), (212, 80, 135),
          (255, 124, 67)]


def prob_viz(res, signs, input_frame, colors):
    """To visualize the detection probability"""
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60+num*40),
                      (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, signs[num], (0, 85+num*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return output_frame


# ML Model Creation
model = Sequential()
model.add(LSTM(64, return_sequences=True,
          activation='relu', input_shape=(30, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(signs.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

# model.load_weights('./models/test_slr.h5')
model = tf.keras.models.load_model('./models/slr.keras')


# 1. New detection variables
sequence = []
sentence = []
threshold = 0.8

cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Make detections
        image, results = utils.mediapipe_detection(frame, holistic)

        # Draw landmarks
        utils.draw_styled_landmarks(image, results)

        # 2. Prediction logic
        keypoints = utils.extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(signs[np.argmax(res)])

        # 3. Viz logic
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if signs[np.argmax(res)] != sentence[-1]:
                        sentence.append(signs[np.argmax(res)])
                else:
                    sentence.append(signs[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, signs, image, colors)

        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('SLR Demo', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
