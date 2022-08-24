import pickle

import numpy as np

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# export TF_FORCE_GPU_ALLOW_GROWTH=true

import keras.backend as K

class AdversarialDriving:

    def __init__(self, model, epsilon = 1):

        self.model = model

        # Initialize Image-Specific Attack
        # Get the loss and gradient of the loss wrt the inputs
        self.attack_type = "image_specific_left"
        self.activate = False

        self.loss = K.mean(-self.model.output, axis=-1)
        self.grads = K.gradients(self.loss, self.model.input)

        # Get the sign of the gradient
        self.delta = K.sign(self.grads[0])

        self.sess = tf.compat.v1.keras.backend.get_session()

        self.perturb = 0
        self.perturbs = []
        self.perturb_percent = 0
        self.perturb_percents = []
        self.n_attack = 1

        self.lr = 0.0002
        self.epsilon = epsilon
        self.xi = 4

        self.image_agnostic_right = np.zeros((1, 160, 320, 3))
        self.image_agnostic_left = np.zeros((1, 160, 320, 3))

        self.result = {}

    def init(self, attack_type, activate):

        # Reset Training Process
        if self.attack_type != attack_type:
            self.perturb = 0
            self.perturbs = []
            self.perturb_percent = 0
            self.perturb_percents = []
            self.n_attack = 1

        self.attack_type = attack_type

        if(activate == 1):
            self.activate = True
            print("Attacker:", attack_type)
        else:
            self.activate = False
            print("No Attack")

            # Initialize Image-Specific Attack
            # Get the loss and gradient of the loss wrt the inputs
            if(attack_type == "image_specific_left" or attack_type == "image_agnostic_left" or attack_type == "image_agnostic_left_train"):
                self.loss = -self.model.output
            if(attack_type == "image_specific_right" or attack_type == "image_agnostic_right" or attack_type == "image_agnostic_right_train"):
                self.loss = self.model.output

            self.grads = K.gradients(self.loss, self.model.input)
            # Get the sign of the gradient
            self.delta = K.sign(self.grads[0])

            # Save Universal Adversarial Perturbation
            if(attack_type == "image_agnostic_right_train"):
                pickle.dump(self.image_agnostic_right, open( "image_agnostic_right.pickle", "wb" ) )
            if(attack_type == "image_agnostic_left_train"):
                pickle.dump(self.image_agnostic_left, open( "image_agnostic_left.pickle", "wb" ) )

            print("Initialized", attack_type)

    def set_image_agnostic_right(self, image_agnostic_right):
        self.image_agnostic_right = image_agnostic_right

    def set_image_agnostic_left(self, image_agnostic_left):
        self.image_agnostic_left = image_agnostic_left

    # Deep Fool: Project on the lp ball centered at 0 and of radius r
    def proj_lp(self, v, r=8, p=2):

        # SUPPORTS only p = 2 and p = Inf for now
        if p == 2:
            v = v * min(1, r / np.linalg.norm(v.flatten('C')))
            # v = v / np.linalg.norm(v.flatten(1)) * xi
        elif p == np.inf:
            v = np.sign(v) * np.minimum(abs(v), r)
        else:
            raise ValueError('Values of p different from 2 and Inf are currently not supported...')

        return v

    def attack(self, input):

        # Random Noise
        if self.attack_type == "random":
            # Random Noises [-epsilon, +epsilon]
            noise = (np.random.randint(2, size=(160, 320, 3)) - 1) * self.epsilon

            return noise

        # Apply Image-Specific Attack
        if self.attack_type.startswith("image_specific_"):
            # Perturb the image
            noise = self.epsilon * self.sess.run(self.delta, feed_dict={self.model.input:np.array([input])})
            return noise.reshape(160, 320, 3)

        # Apply Universal Adversarial Perturbation
        if self.attack_type == "image_agnostic_right":
            return self.image_agnostic_right.reshape(160, 320, 3)

        if self.attack_type == "image_agnostic_left":
            return self.image_agnostic_left.reshape(160, 320, 3)

        # Train Universal Perturbation
        if self.attack_type == "image_agnostic_right_train" or self.attack_type == "image_agnostic_left_train":

            image = np.array([input])
            y_true = float(self.model.predict(image, batch_size=1))

            target_sign = 0

            if self.attack_type == "image_agnostic_right_train":
                target_sign = 1
            if self.attack_type == "image_agnostic_left_train":
                target_sign = -1 

            if (np.sign(y_true) != target_sign):
                x_adv = image
                y_h = y_true

                while(np.sign(y_h) != target_sign):
                    # print("Attack: ", y_h)
                    grads_array = self.sess.run(self.grads, feed_dict={self.model.input:np.array(x_adv)})
                    grads_array = np.array(grads_array[0])

                    grads_array = self.xi * grads_array / np.linalg.norm(grads_array.flatten())

                    x_adv = x_adv + grads_array
                    y_h = self.model.predict(x_adv, batch_size=1)

                    # print("After DeepFool: ", y_true, " --> ", y_h)

                if self.attack_type == "image_agnostic_right_train":
                    self.image_agnostic_right = self.image_agnostic_right + self.lr * (x_adv - image) / self.xi
                    # self.lr = self.lr * 0.99

                    # Project on l_p ball
                    self.image_agnostic_right = self.proj_lp(self.image_agnostic_right, r=self.epsilon, p=np.inf)

                    y_uni = self.model.predict(image + self.image_agnostic_right, batch_size=1)

                if self.attack_type == "image_agnostic_left_train":
                    self.image_agnostic_left = self.image_agnostic_left + self.lr * (x_adv - image) / self.xi
                    # self.lr = self.lr * 0.99

                    # Project on l_p ball
                    self.image_agnostic_left = self.proj_lp(self.image_agnostic_left, r=self.epsilon, p=np.inf)

                    y_uni = self.model.predict(image + self.image_agnostic_left, batch_size=1)

                # print("After Universal: ", y_true, " --> ", y_uni)

                self.perturb = float(y_uni - y_true)
                self.perturbs.append(float(y_uni - y_true))

                self.perturb_percent = self.perturb_percent + (y_uni - y_true) / (np.abs(y_true))
                self.perturb_percents.append(float(self.perturb_percent * 100 / self.n_attack))
                self.n_attack = self.n_attack + 1

            return 0
