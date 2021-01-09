from keras.callbacks import Callback
import numpy as np
import theano

class LossHistory(Callback):
    def __init__(self, X_train, y_train, layer_index):
        super(Callback, self).__init__()
        self.layer_index = layer_index
        self.previous_model_params_ = None
        if X_train.shape[0] >= 1000:
            mask = np.random.choice(X_train.shape[0], 1000)
            self.X_train_subset = X_train[mask]
            self.y_train_subset = y_train[mask]
        else:
            self.X_train_subset = X_train
            self.y_train_subset = y_train

    def on_train_begin(self, logs={}):
        #self.model_params would be ['layer1_W', 'layer2_b', 'layer2_W', 'layer2_b',...]
        self.train_batch_loss = []
        self.train_acc = []
        self.val_acc = []
        self.layer_out = []
        self.model_params = []
        self.gradients = []
        self.init_params = self.get_model_params()
        self.previous_model_params_ = self.init_params

    def on_batch_end(self, batch, logs={}):
        self.train_batch_loss.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        #use W' = W + dW to calculate the gradients
        current_model_params = self.get_model_params()
        gradients = [(param - prev_param) for (param, prev_param) in zip(current_model_params,
                                                                         self.previous_model_params_)]
        self.gradients.append(gradients)
        self.previous_model_params_ = current_model_params
        self.model_params.append(current_model_params)

        self.layer_out.append(self.get_layer_out())

        val_epoch_acc = logs.get('val_acc')
        self.val_acc.append(val_epoch_acc)
        train_epoch_acc = self.model.evaluate(self.X_train_subset, self.y_train_subset,
                                              show_accuracy=True, verbose=0)[1]
        self.train_acc.append(train_epoch_acc)
        print('(train accuracy, val accuracy): (%.4f, %.4f)' % (train_epoch_acc, val_epoch_acc))

    def get_layer_out(self):
        layer_index = self.layer_index
        get_activation = theano.function([self.model.layers[0].input], self.model.layers[layer_index].get_output(train=False), allow_input_downcast=True)
        return get_activation(self.X_train_subset)

    def get_model_params(self):
        return [param.get_value() for param in self.model.params]