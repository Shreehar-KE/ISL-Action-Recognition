import cv2
import numpy as np
import os
import time
import utils
from utils import BreakIt


DATA_PATH = utils.DATA_PATH
no_sequences = 30  # no of clips per sign
sequence_length = 30  # no of frames per clip


signs = utils.signs


mp_holistic = utils.mp_holistic
mp_drawing = utils.mp_drawing


def create_folders(signs, no_sequences):
    """To create folder for each new sign"""
    for sign in signs:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH, sign, str(sequence)))
            except:
                pass


def check_name(name):
    """To check whether a sign already exists"""
    if name in signs:
        print(f'Sign for {name} already exists...!')
        return False
    else:
        return True


def capture_signs(signs):
    """To capture clips for sign(s) using OpenCV"""
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
                        image, results = utils.mediapipe_detection(
                            frame, holistic)

                        # Draw landmarks
                        utils.draw_styled_landmarks(image, results)

                        # Text & Wait logic
                        if frame_num == 0:
                            cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image, 'Collecting frames for {} sequence {}'.format(sign, (sequence+1)), (15, 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(1000)
                        else:
                            cv2.putText(image, 'Collecting frames for {} sequence {}'.format(sign, (sequence+1)), (15, 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)

                        # Export keypoints
                        keypoints = utils.extract_keypoints(results)
                        npy_path = os.path.join(
                            DATA_PATH, sign, str(sequence), str(frame_num))
                        np.save(npy_path, keypoints)

                        # Break gracefully
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            raise BreakIt
            cap.release()
            cv2.destroyAllWindows()
    except BreakIt:
        cap.release()
        cv2.destroyAllWindows()
        pass


def preprocess():
    """To preprocess data & create labels and features"""
    signs = utils.signs
    print(signs)
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

    # print(np.array(sequences).shape)
    # print(np.array(labels).shape)
    np.save('sequences', sequences)
    np.save('labels', labels)


# Controller
while True:
    try:
        print('1.Add Sign\n2.Preprocess Data\n3.Quit')
        ch = int(input('--> '))
        if ch == 1:
            temp = []

            no_of_signs = int(input('Enter the no. of signs to be added--> '))
            for i in range(no_of_signs):
                sign_name = input(f'Enter the name of sign {(i+1)}--> ')
                if check_name(sign_name):
                    temp.append(sign_name)

            ip_signs = np.array(temp)
            no_sequences = 30
            sequence_length = 30

            create_folders(ip_signs, no_sequences)

            print(
                '\n*********** Starting SIGN INPUT COLLECTION for in 5 secs ***********\n')
            time.sleep(5)
            capture_signs(ip_signs)

        elif ch == 2:
            if len(signs) > 0:
                preprocess()
            else:
                print('\n*********** No Signs are added yet ***********\n')
                pass

        elif ch == 3:
            break
        else:
            print('\n*********** WRONG INPUT, Try Again ***********\n')
            continue
    except:
        print('\n*********** WRONG INPUT, Try Again ***********\n')
