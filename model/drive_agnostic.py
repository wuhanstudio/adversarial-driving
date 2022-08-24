import os
import utils
import pickle
import shutil
import argparse
import glob
from datetime import datetime

import cv2

# Deep Learning Libraries
import numpy as np
np.set_printoptions(suppress=True)
from keras.models import load_model

from adversarial_driving import AdversarialDriving

from logger import TensorBoardLogger

# Tensorboard
log_dir = 'logs/image-agnostic/train/hyper/1_0.0004_4/' + datetime.now().strftime("%Y%m%d-%H%M%S")
tb = TensorBoardLogger(log_dir)

EPSILON = 1

N_ITERATION = 500

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Adversarial Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image',
        type=str,
        nargs='?',
        default='data/IMG/',
        help='Path to the driving record image folder.'
    )
    parser.add_argument(
        'attack',
        type=str,
        nargs='?',
        default='right',
        help='The target direction of the attack.'
    )
    args = parser.parse_args()

    log_count = 0

    # Load Images
    image_list = []
    for i, filename in enumerate(glob.glob(args.image +'/center*.jpg')): #assuming gif
        if (i % 10 == 0):
            image = cv2.imread(filename)
            image_list.append(image)

    # Load model
    model = load_model(args.model)
    model.summary()

    # Initialize Adversarial Driving
    adv_drv = AdversarialDriving(model, epsilon = EPSILON)

    # Initialize Attack
    if args.attack == "right":
        adv_drv.init("image_agnostic_right_train", 0)
        adv_drv.init("image_agnostic_right_train", 1)

    elif args.attack == "left":
        adv_drv.init("image_agnostic_left_train", 0)
        adv_drv.init("image_agnostic_left_train", 1)
    else:
        print("Invalid direction (left / right)")
        quit()

    for n in range(N_ITERATION):
        print("Iteration", n)
        perturb_list = []
        for i, image in enumerate(image_list):

            # cv2.imshow("Original", image)
            # cv2.waitKey(1)

            image = utils.preprocess(image) # apply the preprocessing

            perturb = adv_drv.attack(image)

            if(len(adv_drv.perturbs) > 0 and len(adv_drv.perturb_percents) > 0):
                perturb_list.append(adv_drv.perturbs[-1])
                tb.log_scalar('y_uni', float(adv_drv.perturbs[-1]), log_count)

            # Update the log count
            log_count = log_count + 1

        tb.log_scalar('y_abs', np.mean(np.abs(np.array(perturb_list))), n)
        tb.log_scalar('y_mean', np.mean(np.array(perturb_list)), n)
        print("Average:", np.mean(np.array(perturb_list)))

    # Save the Perturbation
    if args.attack == "right":
        adv_drv.init("image_agnostic_right_train", 0)

    elif args.attack == "left":
        adv_drv.init("image_agnostic_left_train", 0)
