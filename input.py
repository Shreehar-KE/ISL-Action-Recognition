import cv2
import numpy as np
import os
import time
import mediapipe as mp


DATA_PATH = os.path.join('MP_Data')
no_sequences = 30
sequence_length = 30
signs = [name for name in os.listdir('./'+DATA_PATH)]
signs = np.array(signs)


class BreakIt(Exception):
    pass


def mediapipe_detection(image, model):
    # COLOR CONVERSION BGR 2 RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable

    # COLOR CONVERSION RGB 2 BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image, results


def draw_styled_landmarks(image, results):
    # Draw Face Connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(
                                  color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

    # Draw Pose Connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))

    # Draw Left Hand Connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2))

    # Draw Right Hand Connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))


mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten(
    ) if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
    ) if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


def create_folders(signs, no_sequences):
    for sign in signs:
        # dirmax = np.max(
        #     np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int))
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, sign, str(sequence)))
            except:
                pass


def check_name(name):
    if name in signs:
        print(f'Sign for {name} already exists...!')
        return False
    else:
        return True


def capture_signs():
    cap = cv2.VideoCapture(0)
    try:
        # Set mediapipe model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

            for sign in signs:
                for sequence in range(no_sequences):
                    for frame_num in range(sequence_length):

                        # Read feed
                        ret, frame = cap.read()
                        frame = cv2.flip(frame, 1)

                        # Make detections
                        image, results = mediapipe_detection(frame, holistic)

                        # Draw landmarks
                        draw_styled_landmarks(image, results)

                        # wait logic
                        if frame_num == 0:
                            cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image, 'Collecting frames for {} sequence {}'.format(sign, (sequence+1)), (15, 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(2000)
                        else:
                            cv2.putText(image, 'Collecting frames for {} sequence {}'.format(sign, (sequence+1)), (15, 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)

                        # NEW Export keypoints
                        keypoints = extract_keypoints(results)
                        npy_path = os.path.join(
                            DATA_PATH, sign, str(sequence), str(frame_num))
                        np.save(npy_path, keypoints)

                        # Break gracefully
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            raise BreakIt
    except BreakIt:
        pass


def preprocess():
    label_map = {label: num for num, label in enumerate(signs)}

    sequences, labels = [], []
    for sign in signs:
        for sequence in range(no_sequences):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, sign, str(
                    sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[sign])
    print(np.array(sequences).shape)
    print(np.array(labels).shape)
    np.save('sequences', sequences)
    np.save('labels', labels)


print('1.Add Sign\n2.Preprocess Data')
ch = int(input('--> '))
if ch == 1:
    temp = []

    no_of_signs = int(input('Enter the no. of signs to be added--> '))
    for i in range(no_of_signs):
        sign_name = input(f'Enter the name of sign {(i+1)}--> ')
        if check_name(sign_name):
            temp.append(sign_name)

    signs = np.array(temp)
    no_sequences = 30
    sequence_length = 30

    create_folders(signs, no_sequences)
    print('\n*********** Starting SIGN INPUT COLLECTION in 5 secs ***********\n')
    time.sleep(5)
    capture_signs()

elif ch == 2:
    if len(signs) > 0:
        preprocess()
    else:
        pass
