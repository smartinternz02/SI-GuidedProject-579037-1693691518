# Import Libraries
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, Response, render_template

# Load the trained model
model = load_model('asl_model.h5')

# Flask App
app = Flask(__name__)

# Video Class
class Video(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.roi_start = (50, 150)
        self.roi_end = (250, 350)
        self.index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        self.y = None
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        ret, frame = self.video.read()
        frame = cv2.resize(frame, (640, 480))
        copy = frame.copy()
        copy = copy[150:150 + 200, 50:50 + 200]

        # Prediction Start
        cv2.imwrite('image.jpg', copy)
        copy_img = image.load_img('image.jpg', target_size=(64, 64))
        x = image.img_to_array(copy_img)
        x = np.expand_dims(x, axis=0)
        pred = np.argmax(model.predict(x), axis=1)
        self.y = pred[0]
        cv2.putText(frame, 'The Predicted Alphabet is: ' + str(self.index[self.y]), (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/index2')
def index2():
    return render_template('index2.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    video = Video()
    return Response(gen(video), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask App
if __name__ == '__main__':
    app.run()
