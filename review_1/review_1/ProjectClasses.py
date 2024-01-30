import cv2
import numpy as np
import mediapipe as mp
import csv
import pandas as pd
import pyttsx3
from time import sleep
from sklearn.neighbors import KNeighborsClassifier
import datetime
#-----------------------------------------------------------------------------------------------------------------------

class frameworking:
    def report(self,yoganame,result):
        time = [datetime.datetime.now()]
        data = {            'DATE & TIME':time,
            'POSE NAME':[yoganame],
            'OUTPUT/RESULTS':[result]
        }
        df = pd.DataFrame(data)
        df.to_csv('report.csv',mode='a',index=False,header=False)
        self.speak(result)
    def speak(self,text):
        engine = pyttsx3.init('sapi5')
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[0].id)
        engine.setProperty('rate',178)
        engine.say(text)
        engine.runAndWait()
        print(text)

    # -----------------------------------------------------------------------------------------------------------------------

    def calculate_angle(self,a, b, c):
        a = np.array(a)  # first
        b = np.array(b)  # mid
        c = np.array(c)  # last
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    # -----------------------------------------------------------------------------------------------------------------------

    def pose_landmarks(self,image,mp_pose):

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            result = pose.process(image)
            pose_landmarks = result.pose_landmarks
            pose_landmarks = [[lmk.x, lmk.y] for lmk in pose_landmarks.landmark]

            return pose_landmarks

    # -----------------------------------------------------------------------------------------------------------------------

    def gen_frames(self,yoganame="NONE"):
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        camera = cv2.VideoCapture(0)
        self.speak("The selected yoga is "+yoganame+".")
        sleep(10)
        fullimage = []
        i= 0
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while i<50:
                success, frame = camera.read()  # read the camera frame
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if not success:
                    break
                else:
                    ret, buffer = cv2.imencode('.jpg', image)
                    fullimage.append(image)
                    frame = image
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpg\r\n\r\n' + frame + b'\r\n')
                    i = i + 1
        camera.release()
        if yoganame=="LUNGES":
            self.lunges(mp_pose, fullimage,yoganame)
        elif yoganame=="TRIKONASANA":
            self.trikonasana(mp_pose,fullimage,yoganame)
        elif yoganame=="ARDHA UTTASANA":
            self.ardha_uttasana(mp_pose,fullimage,yoganame)
        elif yoganame=="VYAGHRASANA":
            self.Vyaghrasana(mp_pose,fullimage,yoganame)
        elif yoganame=="ARDHA CHANDRASANA":
            self.ardha_chandrasana(mp_pose,fullimage,yoganame)
        elif yoganame=="PLANK":
            self.planks(mp_pose,fullimage,yoganame)
        else:
            self.speak("Currently the entered pose is not in the list try to do the pose in the option")

    # -----------------------------------------------------------------------------------------------------------------------

    def lunges(self,mp_pose,fullimage,yoganame):
        fullvalues = []
        for i in range(len(fullimage)):
            try:
                pose_landmarks = self.pose_landmarks(fullimage[i],mp_pose)
                # get cordinates
                Lankel = pose_landmarks[27]
                Lknee = pose_landmarks[25]
                Lhip = pose_landmarks[23]

                Rankel = pose_landmarks[28]
                Rknee = pose_landmarks[26]
                Rhip = pose_landmarks[24]
                        # calculate angle
                left_side_angle = self.calculate_angle(Lankel, Lknee, Lhip)
                right_side_angle = self.calculate_angle(Rankel, Rknee, Rhip)
                fullvalues.append([left_side_angle,right_side_angle])
            except:
                print("error")
        if len(fullvalues)>20:
            with open('common.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['left_leg','right_leg'])
                writer.writerows(fullvalues)

            names = ['left_leg', 'right_leg', 'results']
            from sklearn.neighbors import KNeighborsClassifier
            classifier = KNeighborsClassifier(n_neighbors=3)
            data = pd.read_csv("csv/lunges.csv", names=names)
            X = data.iloc[1:, :-1].values
            y = data.iloc[1:, -1].values
            video_data = pd.read_csv("common.csv")
            video_data_X = video_data.iloc[1:, :].values
            classifier.fit(X, y)
            list_data_video_output = classifier.predict(video_data_X)
            output = max(list_data_video_output)
            self.report(yoganame, output)
        else:
            self.speak("YOU ARE OUT OF FRAME! OR DO IN GOOD LIGHT CONDITION")
# -----------------------------------------------------------------------------------------------------------------------

    def trikonasana(self,mp_pose,fullimage,yoganame):
        fullvalues = []
        for i in range(len(fullimage)):
            try:
                pose_landmarks = self.pose_landmarks(fullimage[i],mp_pose)
                # get cordinates
                left_shoulder_angle = self.calculate_angle(pose_landmarks[15], pose_landmarks[11], pose_landmarks[23])
                right_shoulder_angle = self.calculate_angle(pose_landmarks[16], pose_landmarks[12], pose_landmarks[24])
                left_hip_angle = self.calculate_angle(pose_landmarks[11], pose_landmarks[23], pose_landmarks[27])
                right_hip_angle = self.calculate_angle(pose_landmarks[12], pose_landmarks[24], pose_landmarks[28])
                fullvalues.append([left_shoulder_angle,right_shoulder_angle,left_hip_angle,right_hip_angle])
            except:
                print("error")
        if len(fullvalues)>20:
            with open('common.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['left_shoulder','right_shoulder','left_hip','right_hip'])
                writer.writerows(fullvalues)

            names = ['left_shoulder','right_shoulder','left_hip','right_hip','results']
            from sklearn.neighbors import KNeighborsClassifier
            classifier = KNeighborsClassifier(n_neighbors=3)
            data = pd.read_csv("csv/trikonasana.csv", names=names)
            X = data.iloc[1:, :-1].values
            y = data.iloc[1:, -1].values
            video_data = pd.read_csv("common.csv")
            video_data_X = video_data.iloc[1:, :].values
            classifier.fit(X, y)
            list_data_video_output = classifier.predict(video_data_X)
            output = max(list_data_video_output)
            self.report(yoganame, output)
        else:
            self.speak("YOU ARE OUT OF FRAME! OR DO IN GOOD LIGHT CONDITION")

    # -------------------------------------------------------------------------------------------------------------------

    def ardha_uttasana(self,mp_pose,fullimage,yoganame):
        fullvalues = []
        for i in range(len(fullimage)):
            try:
                pose_landmarks = self.pose_landmarks(fullimage[i],mp_pose)
                # get cordinates
                left_hip = self.calculate_angle(pose_landmarks[11], pose_landmarks[23], pose_landmarks[27])
                right_hip = self.calculate_angle(pose_landmarks[12], pose_landmarks[24], pose_landmarks[28])
                left_knee = self.calculate_angle(pose_landmarks[23], pose_landmarks[25], pose_landmarks[27])
                right_knee = self.calculate_angle(pose_landmarks[24], pose_landmarks[26], pose_landmarks[28])
                fullvalues.append([left_hip,right_hip,left_knee,right_knee])
            except:
                print("error")
        if len(fullvalues)>20:
            with open('common.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['left_hip','right_hip','left_knee','right_knee'])
                writer.writerows(fullvalues)

            names = ['left_hip','right_hip','left_knee','right_knee','results']
            classifier = KNeighborsClassifier(n_neighbors=3)
            data = pd.read_csv("csv/ardha_uttasana.csv", names=names)
            X = data.iloc[1:, :-1].values
            y = data.iloc[1:, -1].values
            video_data = pd.read_csv("common.csv")
            video_data_X = video_data.iloc[1:, :].values
            classifier.fit(X, y)
            list_data_video_output = classifier.predict(video_data_X)
            output = max(list_data_video_output)
            self.report(yoganame, output)
        else:
            self.speak("YOU ARE OUT OF FRAME! OR DO IN GOOD LIGHT CONDITION")
    #-----------------------------------------------------------------------------------------------------------------------

    def Vyaghrasana(self,mp_pose,fullimage,yoganame):
        fullvalues = []
        for i in range(len(fullimage)):
            try:
                pose_landmarks = self.pose_landmarks(fullimage[i],mp_pose)
                # get cordinates
                lefthand = self.calculate_angle(pose_landmarks[11], pose_landmarks[13], pose_landmarks[15])
                righthand = self.calculate_angle(pose_landmarks[12], pose_landmarks[14], pose_landmarks[16])
                leftleg = self.calculate_angle(pose_landmarks[23], pose_landmarks[25], pose_landmarks[27])
                rightleg = self.calculate_angle(pose_landmarks[24], pose_landmarks[26], pose_landmarks[28])
                leftshouder = self.calculate_angle(pose_landmarks[13], pose_landmarks[11], pose_landmarks[23])
                rightshouder = self.calculate_angle(pose_landmarks[14], pose_landmarks[12], pose_landmarks[24])

                fullvalues.append([lefthand,righthand,leftleg,rightleg,leftshouder,rightshouder])
            except:
                print("error")
        if len(fullvalues)>20:
            with open('common.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['lefthand','righthand','leftleg','rightleg','leftshouder','rightshouder'])
                writer.writerows(fullvalues)

            names = ['lefthand','righthand','leftleg','rightleg','leftshouder','rightshouder','results']
            classifier = KNeighborsClassifier(n_neighbors=3)
            data = pd.read_csv("csv/vyaghrasana.csv", names=names)
            X = data.iloc[1:, :-1].values
            y = data.iloc[1:, -1].values
            video_data = pd.read_csv("common.csv")
            video_data_X = video_data.iloc[1:, :].values
            classifier.fit(X, y)
            list_data_video_output = classifier.predict(video_data_X)
            output = max(list_data_video_output)
            self.report(yoganame, output)
        else:
            self.speak("YOU ARE OUT OF FRAME! OR DO IN GOOD LIGHT CONDITION")
    #-----------------------------------------------------------------------------------------------------------------------

    def ardha_chandrasana(self,mp_pose,fullimage,yoganame):
        fullvalues = []
        for i in range(len(fullimage)):
            try:
                pose_landmarks = self.pose_landmarks(fullimage[i],mp_pose)
                # get cordinates
                lefthand = self.calculate_angle(pose_landmarks[11], pose_landmarks[13], pose_landmarks[15])
                righthand = self.calculate_angle(pose_landmarks[12], pose_landmarks[14], pose_landmarks[16])
                leftleg = self.calculate_angle(pose_landmarks[23], pose_landmarks[25], pose_landmarks[27])
                rightleg = self.calculate_angle(pose_landmarks[24], pose_landmarks[26], pose_landmarks[28])
                leftshouder = self.calculate_angle(pose_landmarks[13], pose_landmarks[11], pose_landmarks[23])
                rightshouder = self.calculate_angle(pose_landmarks[14], pose_landmarks[12], pose_landmarks[24])

                fullvalues.append([lefthand,righthand,leftleg,rightleg,leftshouder,rightshouder])
            except:
                print("error")
        if len(fullvalues)>20:
            with open('common.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['lefthand','righthand','leftleg','rightleg','leftshouder','rightshouder'])
                writer.writerows(fullvalues)

            names = ['lefthand','righthand','leftleg','rightleg','leftshouder','rightshouder','results']
            classifier = KNeighborsClassifier(n_neighbors=3)
            data = pd.read_csv("csv/ardha chandrasana.csv", names=names)
            X = data.iloc[1:, :-1].values
            y = data.iloc[1:, -1].values
            video_data = pd.read_csv("common.csv")
            video_data_X = video_data.iloc[1:, :].values
            classifier.fit(X, y)
            list_data_video_output = classifier.predict(video_data_X)
            output = max(list_data_video_output)
            self.report(yoganame, output)
        else:
            self.speak("YOU ARE OUT OF FRAME! OR DO IN GOOD LIGHT CONDITION")


    def planks(self,mp_pose,fullimage,yoganame):
        fullvalues = []
        marks = [[16, 14, 12], [14, 12, 24], [12, 24, 26], [24, 26, 28], [14, 12, 26], [14, 12, 28], [12, 24, 28],
                 [12, 26, 28], [16, 12, 24], [16, 12, 26], [16, 12, 28],
                 [15, 13, 11], [13, 11, 23], [11, 23, 25], [23, 25, 27], [13, 11, 25], [13, 11, 27], [11, 23, 27],
                 [11, 25, 27], [15, 11, 23], [15, 11, 25], [15, 11, 27]]

        for i in range(len(fullimage)):
            angles = []
            try:
                pose_landmarks = self.pose_landmarks(fullimage[i],mp_pose)
                # get cordinates
                for mark in range(0, len(marks)):
                    angles.append(self.calculate_angle(pose_landmarks[marks[mark][0]], pose_landmarks[marks[mark][1]],
                                                        pose_landmarks[marks[mark][2]]))

                fullvalues.append(angles)
            except:
                print("error")
        if len(fullvalues)>20:
            with open('common.csv', 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21'])
                writer.writerows(fullvalues)

            names = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','results']
            classifier = KNeighborsClassifier(n_neighbors=3)
            data = pd.read_csv("csv/planks.csv", names=names)
            X = data.iloc[1:, :-1].values
            y = data.iloc[1:, -1].values
            video_data = pd.read_csv("common.csv")
            video_data_X = video_data.iloc[1:, :].values
            classifier.fit(X, y)
            list_data_video_output = classifier.predict(video_data_X)
            output = max(list_data_video_output)
            self.report(yoganame, output)
        else:
            self.speak("YOU ARE OUT OF FRAME! OR DO IN GOOD LIGHT CONDITION")