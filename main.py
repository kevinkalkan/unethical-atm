from flask import Flask, render_template
from teachable_machine import TeachableMachine
import cv2 as cv
import time

app = Flask(__name__)

model = TeachableMachine(model_path="keras_model.h5",
                         labels_file_path="labels.txt")

cap = cv.VideoCapture(0)

def start_video_capture():
    _, img = cap.read()
    timestamp = int(time.time())
    image_path = f"screenshot_{timestamp}.jpg"
    cv.imwrite(image_path, img)
    result = model.classify_image(image_path)
    print("class_index", result["class_index"])
    print("class_name:::", result["class_name"])
    return result["class_name"]

# Render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['GET'])
def classify():
    global stored_class_name
    stored_class_name = start_video_capture()
    return stored_class_name
    
@app.route('/stop')
def stop():
    cv.destroyAllWindows()
    return 'Video stream stopped.'

if __name__ == '__main__':
    app.run(debug=True)
