from keras.layers import Dense, Layer, InputSpec
import keras.backend as K
from keras import initializations, regularizers


def level1_attention(layers=None, name=None, node_indices=None, tensor_indices=None):
    all_keras_tensors = True
    for x in layers:
        if not hasattr(x, '_keras_history'):
            all_keras_tensors = False
            break
    if all_keras_tensors:
        input_layers = []
        node_indices = []
        tensor_indices = []
        for x in layers:
            input_layer, node_index, tensor_index = x._keras_history
            input_layers.append(input_layer)
            node_indices.append(node_index)
            tensor_indices.append(tensor_index)
        layer = Level1AttentionLayer(input_layers, node_indices=node_indices, tensor_indices=tensor_indices)
        return layer.inbound_nodes[0].output_tensors[0]
    else:
        node_indices = node_indices
        input_layers = layers
        tensor_indices = tensor_indices
        return Level1AttentionLayer(input_layers, node_indices, tensor_indices)


    # our layer will take input shape (nb_samples, 1)
class Level1AttentionLayer(Layer):
    def __init__(self, layers=None, name=None, node_indices=None, tensor_indices=None, output_shape=None, output_mask=None, **kwargs):
        self._output_shape = output_shape
        self._output_mask = output_mask

        self.inbound_nodes = []
        self.outbound_nodes = []
        self.constraints = {}
        self.regularizers = []
        self.trainable_weights = []
        self.non_trainable_weights = []
        self.supports_masking = False
        self.uses_learning_phase = False
        self.input_spec = None  # compatible with whatever
        self.input_dim = None

        self.input_layers = layers
        self.node_indices = node_indices
        self.tensor_indices = tensor_indices
        self.trainable = False
        if not name:
            prefix = self.__class__.__name__.lower()
            name = prefix + '_' + str(K.get_uid(prefix))
        self.name = name
        if layers:
            # this exists for backwards compatibility.
            # equivalent to:
            # merge = Merge(layers=None)
            # output = merge([input_tensor_1, input_tensor_2])
            if not node_indices:
                # by default we connect to
                # the 1st output stream in the input layer
                node_indices = [0 for _ in range(len(layers))]
            self.built = True
            self.add_inbound_node(self.input_layers, self.node_indices, self.tensor_indices)

        self.trainable_weights = []

    def build(self, input_shape):
        assert type(input_shape) is list and len(input_shape) == 2

    def get_output_shape_for(self, input_shape):
        # we're doing a scalar multiply, so we don't change the input shape
        assert type(input_shape) is list and len(input_shape) == 2

        s0 = input_shape[0]
        s1 = input_shape[1]
        return s0[0], s0[1], s1[1]

    def call(self, inputs, mask=None):
        #works only for 2 inputs
        if type(inputs) is not list or not len(inputs) == 2:
            raise Exception('Merge must be called on a list of tensors '
                            '(at least 2). Got: ' + str(inputs))

        X = inputs[0]
        Y = inputs[1]

        x_square = K.square(X)
        y_square = K.square(Y)

        x_sq_sum = K.sum(x_square, axis=-1)
        y_sq_sum = K.sum(y_square, axis=-1)

        x_epxa = K.expand_dims(x_sq_sum, dim=-1)
        y_expa = K.expand_dims(y_sq_sum, dim=-2)

        #x_rep = K.repeat(x_sq_sum, n=y_square.shape[1])
        #y_rep = K.repeat(y_sq_sum, n=x_square.shape[1])

        dot = K.batch_dot(X, K.permute_dimensions(Y, (0, 2, 1)), axes=(2, 1))
        #dot = K.dot(X, K.permute_dimensions(Y, (0, 2, 1)))

        sum = K.maximum(x_epxa + y_expa - 2 * dot, K.epsilon())
        #squared_euclidean_similarity = 1/(1 + K.sqrt(K.permute_dimensions(x_sq_sum, (0, 2, 1)) + y_sq_sum - 2 * dot))
        squared_euclidean_similarity = 1/(1 + K.sqrt(sum))

        return squared_euclidean_similarity


class Level1AttentionTransformation(Layer):
    def __init__(self, transpose, output_dim, weights=None, W_regularizer=None, init='glorot_uniform', **kwargs):
        self.init = initializations.get(init)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.initial_weights = weights
        self.transpose = transpose
        self.output_dim = output_dim
        self.trainable = True
        self.input_dim = None
        super(Level1AttentionTransformation, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        if self.transpose:
            input_dim = input_shape[1]
        else:
            input_dim = input_shape[2]

        self.input_spec = [InputSpec(dtype=K.floatx(), shape=input_shape)]

        self.W = self.init((input_dim, self.output_dim), name='{}_W'.format(self.name))
        self.trainable_weights = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, A, mask=None):
        if self.transpose:
            output = K.dot(K.permute_dimensions(A, (0, 2, 1)), self.W)
        else:
            output = K.dot(A, self.W)
        return output

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 3
        if self.transpose:
            return input_shape[0], input_shape[2], self.output_dim
        else:
            return input_shape[0], input_shape[1], self.output_dim

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'input_dim': self.input_dim}
        base_config = super(Level1AttentionTransformation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

