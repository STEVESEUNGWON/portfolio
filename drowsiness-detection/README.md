# 🛠️ Face Recognition and Drowsiness Detection Service

## 📄 Project Description

This project provides a service for face recognition and drowsiness detection. It uses the Flask web framework to build a web application and employs OpenCV and dlib libraries for face recognition and drowsiness detection. The system alerts the user when drowsiness is detected during driving and logs user data into an Excel file.

## 🛠 Technologies Used

- Python
- Flask
- OpenCV
- dlib
- pyttsx3
- OpenPyXL
- pandas
- ultralytics YOLO
- Pygame


## 🧪 Key Features and How to Run
**1. Face Recognition Model Training**

Train the face recognition model using LBPH Face Recognizer. The data consists of face image files stored in the static/faces/ directory.

**2. Face Recognition and Data Logging**

Logs the recognized face data into an Excel file, including date, time, and user name.

**3. Drowsiness Detection System**

Calculates the eye aspect ratio (EAR) to detect drowsiness and alerts the user with an alarm sound if drowsiness is detected for a prolonged period.

**4. Drowsiness Detection and Alert System Using YOLOv8**

Uses the YOLOv8 model to detect drowsiness and alerts the user with an alarm sound. If drowsiness is detected multiple times, it alerts with a safety message and alarm.

**5. Running the Flask Application**

Run the Flask web application to provide real-time face recognition and drowsiness detection. The web page displays the real-time video stream showing the face recognition and drowsiness detection status.

**6.Open the Application in Your Web Browser**

http://127.0.0.1:5000/






**📁 Project Structure**


<img width="200" alt="image" src="https://github.com/STEVESEUNGWON/portfolio/assets/159239472/b7629d53-4d99-494e-948c-8b81454bc407">





## 📸 Screenshots

<img width="500" alt="image" src="https://github.com/STEVESEUNGWON/portfolio/assets/159239472/4094924d-536f-4b24-9fa9-2a1cc9a87f16">
















**Face Recognition System**



<img width="500" alt="image" src="https://github.com/STEVESEUNGWON/portfolio/assets/159239472/98778678-8d16-4296-8f7c-2855d8e8a684">


















**Drowsiness Detection System**

<img width="500" alt="image" src="https://github.com/STEVESEUNGWON/portfolio/assets/159239472/439c4913-6c2b-4b6d-88bd-326a1a93fb23">


<img width="500" alt="image" src="https://github.com/STEVESEUNGWON/portfolio/assets/159239472/01217840-79f6-4ab7-a09f-e6e71c3f800f">










## 🎯 Results
The face recognition and drowsiness detection service monitors the driver's status in real-time and alerts with high accuracy when drowsiness is detected.



## ⚖️ License
This project is licensed under the MIT License. See the LICENSE file for more details.

## 🙏 Acknowledgements
Thanks to Kaggle for providing the dataset.
The data can be adjusted to suit each user's needs.








