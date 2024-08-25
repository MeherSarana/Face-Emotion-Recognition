from deepface import DeepFace
from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from keras.models import model_from_json

app = Flask(__name__)

# Load the pre-trained model architecture from JSON file
json_file = open("E:\\Mini Project\\facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# Load the pre-trained model weights
model.load_weights("E:\\Mini Project\\facialemotionmodel.h5")

# Load the Haar cascade classifier for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Define a function to extract features from an image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Define labels for emotion classes
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def generate_frames():
    # Open the webcam (camera)
    webcam = cv2.VideoCapture(0)

    while True:
        # Read a frame from the webcam
        success, frame = webcam.read()
        if not success:
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) which contains the face
            roi_gray = gray[y:y+w, x:x+h]

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Resize the face image to the required input size (48x48)
            roi_gray = cv2.resize(roi_gray, (48, 48))

            # Extract features from the resized face image
            img = extract_features(roi_gray)

            # Make a prediction using the trained model
            pred = model.predict(img)

            # Get the predicted label for emotion
            prediction_label = labels[pred.argmax()]

            # Display the predicted emotion label near the detected face
            cv2.putText(frame, f'Emotion: {prediction_label}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the output frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release the webcam and close all OpenCV windows
    webcam.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    img_path = "uploaded_image.png"
    file.save(img_path)

    try:
        result = DeepFace.analyze(img_path, actions=['emotion'])
        text = "Emotion Detected: "
        emotion = text + result[0]['dominant_emotion']
    except ValueError:
        emotion = "Face could not be detected in uploaded image."

    return render_template('result.html', emotion=emotion)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
