import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
import warnings

warnings.filterwarnings("ignore")
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# load model
model = load_model("C:\\Users\\KIIT\\PycharmProjects\\emotion&gesturedetection\\fer7_model.h5", compile=False)
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Optional part for writing video
# video_cod = cv2.VideoWriter_fourcc(*'XVID')
# video_output = cv2.VideoWriter('out.mp4',
                               # video_cod,
                               # 10,
                               # (1000, 700))
cap = cv2.VideoCapture(0)
while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255
        predictions = model.predict(img_pixels)
# find max indexed array
        max_index = np.argmax(predictions[0])
        emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        predicted_emotion = emotions[max_index]
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis ', resized_img)

# optional
#         video_output.write(resized_img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
cap.release()
# optional
# video_output.release()
cv2.destroyAllWindows