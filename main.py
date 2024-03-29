from flask import Flask, render_template, redirect, send_file, request, session, url_for
from teachable_machine import TeachableMachine
import time
import cv2 as cv
import os


app = Flask(__name__)
app.static_folder = 'static'
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'
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


@app.route('/survey', methods=['GET', 'POST'])
def survey():
    if request.method == 'POST':
        first_name = request.form.get('fname')
        print("Submitted first name:", first_name)
        
        session['first_name'] = first_name
        return redirect('/terms')

    return render_template('survey.html')


@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/screen')
def screen():
    return render_template('screen.html')

@app.route('/loading')
def loading():
    return render_template('loading.html')

@app.route('/offers')
def offers():
    class_name = start_video_capture() 
    return render_template('offers.html', class_name=class_name)

@app.route('/brown')
def brown():
        
    first_name = session.get('first_name', 'Member')  


    return render_template('card_brown.html', first_name=first_name)

@app.route('/gold')
def gold():
    
    first_name = session.get('first_name', 'Member')  


    return render_template('card_gold.html', first_name=first_name)

@app.route('/pink')
def pink():
    first_name = session.get('first_name', 'Member')
    print("Retrieved from session:", first_name)
    return render_template('card_pink.html', first_name=first_name)

@app.route('/orange')
def orange():
    
    first_name = session.get('first_name', 'Member')  


    return render_template('card_orange.html', first_name=first_name)

@app.route('/green')
def green():
        
    first_name = session.get('first_name', 'Member')  


    return render_template('card_green.html', first_name=first_name)

@app.route('/save_offers')
def save_offers():
    class_name = start_video_capture() 
    html_content = render_template('offers.html', class_name=class_name)
    with open(OFFERS_FILE_PATH, "w", encoding="utf-8") as file:
        file.write(html_content)

    return send_file(OFFERS_FILE_PATH, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
