from flask import Flask, render_template, redirect, send_file
from teachable_machine import TeachableMachine
import time
import cv2 as cv
import os

app = Flask(__name__)
app.static_folder = 'static'

model = TeachableMachine(model_path="keras_model.h5", labels_file_path="labels.txt")

CONSTANT_IMAGE_FILE = "static/screenshot.jpg"
OFFERS_FILE_PATH = "static/offers_output.html"
cap = cv.VideoCapture(0)
time.sleep(1) 
if not cap.isOpened():
    print("Error: Could not open webcam.")

def start_video_capture():
    _, img = cap.read()

    if img is not None and img.any() and img.shape[0] > 0 and img.shape[1] > 0:
        cv.imwrite(CONSTANT_IMAGE_FILE, img)
    # Continue with classification
    else:
        print("Error: Captured frame is empty or has invalid dimensions.")

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
    return redirect('/offers')


@app.route('/survey')
def survey():
    return render_template('survey.html')


@app.route('/screen')
def screen():
    return render_template('screen.html')

@app.route('/loading')
def loading():
    return render_template('loading.html')

@app.route('/offers')
def offers():
    # Assume 'data' is the variable you want to pass to the template
    class_name = start_video_capture()  # Call your function to get the class_name
    return render_template('offers.html', class_name=class_name)

@app.route('/save_offers')
def save_offers():
    class_name = start_video_capture()  # Call your function to get the class_name

    # Render the offers.html template with the class_name
    html_content = render_template('offers.html', class_name=class_name)

    # Save the HTML content to the fixed filename
    with open(OFFERS_FILE_PATH, "w", encoding="utf-8") as file:
        file.write(html_content)

    return send_file(OFFERS_FILE_PATH, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
