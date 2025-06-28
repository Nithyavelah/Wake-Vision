# Wake Vision 👁️🚗

**Wake Vision** is a computer vision-based drowsiness detection system developed for vehicles to identify signs of driver sleepiness. It continuously tracks the driver’s facial features and triggers an audio alert when drowsiness is detected, helping to prevent accidents caused by inattention and promoting road safety.

## 🚀 Features

- 🟢 Real-time drowsiness detection using webcam
- 📹 Analyze pre-recorded videos for eye closure
- 🖼️ Detect eye status in static images
- 🔊 Audio alarm to wake up the driver
- 👁️ Improved detection for side view and partial face visibility

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Playsound
- OS, Time, Threading libraries

## 📁 Project Structure

WakeVision/
│
├── live_webcam_feed.py # Detect drowsiness using webcam
├── pre_recorded_video.py # Analyze pre-recorded video
├── image_detection.py # Analyze eye state in image
├── beep-02.wav # Sound alert for drowsiness- Pre-recorded video
├── alarm_live_feed.pm3 # Sound alert for drwosiness - Live Webcam Feed
├── requirements.txt # Required libraries
└── README.md # Project documentation

## 🧪 How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/WakeVision.git
cd WakeVision
```


### 2. Install Required Libraries
Make sure you have Python installed. Then run the following command to install all required libraries:

```bash
pip install -r requirements.txt
```
### 3. Run the Desired Module

The Wake Vision system offers three modes of drowsiness detection. You can run any of the following based on your input type:


# Run real-time detection using webcam
```
python live_webcam_feed.py
```
# Run detection on a pre-recorded video
```
python pre_recorded_video.py
```
# Run detection on a static image (replace with your image path)
```
python image_detection.py --image path_to_image.jpg
```
## 📸 Screenshots

### ▶️ Real-Time Detection (Webcam)
![Live Detection](screenshots/live_webcam_feed.jpg)

### 🎞️ Video File Analysis
![Video Detection](screenshots/pre_recorded_video.jpg)

### 🖼️ Image Detection
![Image Detection](screenshots/image_detection.jpg)

## 📚 References

- [MediaPipe Face Mesh](https://google.github.io/mediapipe/)
- [OpenCV Documentation](https://docs.opencv.org/)
- Gemini (used to generate sample images for testing)

 ## 👤 Author

**M Nithyavelah*  
GitHub: [@Nithyavelah](https://github.com/nithyavelah)

## 📄 License

This project is licensed under the [MIT License](LICENSE).

 





