from keras.callbacks import Callback
import keras.backend as K


class NaNDetector(Callback):
    def __init__(self):
        super(Callback, self).__init__()
        self.previous_model_params_ = None

    def on_train_begin(self, logs={}):
        #self.model_params would be ['layer1_W', 'layer2_b', 'layer2_W', 'layer2_b',...]
        self.init_params = self.get_model_params()
        self.previous_model_params_ = self.init_params

    def on_batch_end(self, batch, logs={}):
        # use W' = W + dW to calculate the gradients
        current_model_params = self.get_model_params()
        gradients = [(param - prev_param) for (param, prev_param) in zip(current_model_params, self.previous_model_params_)]
        magnitudes = [(param.name, K.eval(K.sqrt(K.sum(K.square(gradient))))) for param, gradient in zip(self.model.trainable_weights, gradients)]

        self.previous_model_params_ = current_model_params
        outline = 'Magnitudes:'
        for name, magnitude in magnitudes:
            outline += '\t{}: {}'.format(name, magnitude)

        print(outline)

    def get_model_params(self):
        return [param.get_value() for param in self.model.trainable_weights]
