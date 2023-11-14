from flask import Flask, render_template, redirect
from teachable_machine import TeachableMachine
import time
import cv2 as cv

app = Flask(__name__)
app.static_folder = 'static'

model = TeachableMachine(model_path="keras_model.h5", labels_file_path="labels.txt")

CONSTANT_IMAGE_FILE = "static/screenshot.jpg"
cap = cv.VideoCapture(0)


def start_video_capture():
    _, img = cap.read()

    # Overwrite the existing screenshot file with the new frame
    cv.imwrite(CONSTANT_IMAGE_FILE, img)

    result = model.classify_image(CONSTANT_IMAGE_FILE)
    class_name = result["class_name"]

    print("class_index", result["class_index"])
    print("class_name:::", class_name)

    return class_name


def stop_video_capture():
    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['GET'])
def classify():
    class_name = start_video_capture()
    return class_name


@app.route('/stop')
def stop():
    stop_video_capture()
    cv.destroyAllWindows()
    return redirect('/')


@app.route('/survey')
def survey():
    return render_template('survey.html')


@app.route('/screen')
def screen():
    return render_template('screen.html')


if __name__ == '__main__':
    app.run(debug=True)
