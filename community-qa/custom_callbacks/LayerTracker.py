from keras.callbacks import Callback
import keras.backend as K
import numpy as np

class LayerTracker(Callback):
    def __init__(self, layer_idx):
        super(Callback, self).__init__()
        self.layer_idx = layer_idx
        self.previous_model_params_ = None

    def on_train_begin(self, logs={}):
        #self.model_params would be ['layer1_W', 'layer2_b', 'layer2_W', 'layer2_b',...]
        self.init_params = self.get_model_layer_params()
        self.previous_model_params_ = self.init_params

        magnitude = K.eval(K.sqrt(K.sum(K.square(self.init_params))))
        outline = 'Magnitudes:\t{}: {}'.format(self.layer_idx, magnitude)
        print (outline)

    def on_batch_end(self, batch, logs={}):
        # use W' = W + dW to calculate the gradients
        current_model_params = self.get_model_layer_params()
        gradient = current_model_params - self.previous_model_params_
        gr = np.nan_to_num(K.eval(gradient))
        mag = np.sqrt(np.sum(np.square(gr)))
        magnitude = K.eval(K.sqrt(K.sum(K.square(gradient))))

        self.previous_model_params_ = current_model_params
        outline = 'Magnitudes:\t{}: {}'.format(self.layer_idx, mag)
        print(outline)

    def get_model_params(self):
        return [param.get_value() for param in self.model.trainable_weights]

    def get_model_layer_params(self):
        return self.model.trainable_weights[self.layer_idx]
