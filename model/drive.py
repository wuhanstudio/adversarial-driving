import os
import utils
import pickle
import shutil
import argparse
from datetime import datetime

# Web framework
import socketio
import eventlet
import eventlet.wsgi

from flask import Flask
from flask_cors import CORS

# Image Processing
from PIL import Image

# Data IO and Encoding-Decoding
from io import BytesIO
import base64

# Deep Learning Libraries
import numpy as np
np.set_printoptions(suppress=True)
from keras.models import load_model

from adversarial_driving import AdversarialDriving

# Initialize the server
sio = socketio.Server(cors_allowed_origins='*')

# Initialize the flask (web) app
app = Flask(__name__)
CORS(app)

# Set min/max speed for our autonomous car
MAX_SPEED = 25
MIN_SPEED = 10

# Speed limit
speed_limit = MAX_SPEED

# From image to base64 string
def img2base64(image):

    origin_img = Image.fromarray(np.uint8(image))
    origin_buff = BytesIO()
    origin_img.save(origin_buff, format="JPEG")

    return base64.b64encode(origin_buff.getvalue()).decode("utf-8")


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


# Registering event handler for the server
@sio.on('attack')
def onAttack(sid, data):
    # Initialize Attack
    adv_drv.init(data["type"], int(data["attack"]))


# Registering event handler for each frame
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        sio.emit('update', {'data': data["image"], 'speed' : data["speed"]})
        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        try:
            image = np.asarray(image)       # from PIL image to numpy array
            image = utils.preprocess(image) # apply the preprocessing

            sio.emit('input', {'data': img2base64(image)})

            y_true = model.predict(np.array([image]), batch_size=1)

            # If the attack is activated
            if adv_drv.activate:
                perturb = adv_drv.attack(image)

                if adv_drv.attack_type == "unir_no_left_train" or adv_drv.attack_type == "unir_no_right_train":
                    if(len(adv_drv.perturbs) > 0 and len(adv_drv.perturb_percents) > 0):
                        sio.emit('unir_train', { 'absolute': str(adv_drv.perturbs[-1]), 'percentage': str(adv_drv.perturb_percents[-1])})
                        # print("Attack Strength: ", float(perturb / n_attack), " Attack Percent: ", perturb_percent * 100 / n_attack, "%")
                    image = np.array([image])
                else:
                    x_adv = np.array(image) + perturb
                    sio.emit('adv', {'data': img2base64(x_adv)})
                    sio.emit('diff', {'data': img2base64(perturb)})

                    image = np.array([x_adv])
                    y_adv = float(model.predict(image, batch_size=1))
                    sio.emit('res', {'original': str(float(y_true)), 'result': str(float(y_adv)),  'percentage': str(float(((y_true-y_adv) * 100 / np.abs(y_true)))) })

                    if adv_drv.attack_type not in adv_drv.result:
                        adv_drv.result[adv_drv.attack_type] = []
                    if(len(adv_drv.result[adv_drv.attack_type]) > 0 and len(adv_drv.result[adv_drv.attack_type]) % 100 == 0):
                        print(adv_drv.attack_type, len(adv_drv.result[adv_drv.attack_type]), np.mean(adv_drv.result[adv_drv.attack_type], axis=0))
                        sio.emit('info', {'type': str(adv_drv.attack_type), 'times': str(len(adv_drv.result[adv_drv.attack_type])), 'res': str(np.mean(adv_drv.result[adv_drv.attack_type], axis=0))})
                    adv_drv.result[adv_drv.attack_type].append([float(np.abs(y_adv-y_true)), float((np.abs(y_true-y_adv) * 100 / np.abs(y_true)))])
            else:
                image = np.array([image])       # the model expects 4D array

            # predict the steering angle for the image
            steering_angle = float(model.predict(image, batch_size=1))
            # lower the throttle as the speed perturbreases
            # if the speed is above the current speed limit, we are on a downhill.
            # make sure we slow down first and then go back to the original max speed.
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # slow down
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

            print('{} {} {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        sio.emit('manual', data={}, skip_sid=True)


# Send control command to the simulater
def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    # Initialize Adversarial Driving
    adv_drv = AdversarialDriving(model)

    if(os.path.isfile("unir_no_left.pickle")):
        with open('unir_no_left.pickle', 'rb') as f:
            adv_drv.set_unir_no_left(pickle.load(f))
    
    if(os.path.isfile("unir_no_right.pickle")):
        with open('unir_no_right.pickle', 'rb') as f:
            adv_drv.set_unir_no_right(pickle.load(f))

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
