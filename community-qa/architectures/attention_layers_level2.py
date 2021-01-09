from keras.layers import Layer
import keras.backend as K
from theano.tensor.signal.pool import pool_2d


def level2_attention(window_size, axes, layers=None, node_indices=None, tensor_indices=None):
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
        layer = Level2AttentionLayer(window_size, axes, input_layers, node_indices=node_indices, tensor_indices=tensor_indices)
        return layer.inbound_nodes[0].output_tensors[0]
    else:
        node_indices = node_indices
        input_layers = layers
        tensor_indices = tensor_indices
        return Level2AttentionLayer(input_layers, node_indices, tensor_indices)


# our layer will take input shape (nb_samples, 1)
class Level2AttentionLayer(Layer):
    def __init__(self, window_size, axes, layers=None, name=None, node_indices=None, tensor_indices=None, output_shape=None, output_mask=None, **kwargs):
        self.window_size = window_size
        self.shape_axes = axes

        if axes == 1:
            self.transform_axis = 2
        elif axes == 2:
            self.transform_axis = 1
        else:
            #TODO maybe throw error
            self.transform_axis = axes


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
        F = input_shape[0]
        A = input_shape[1]

        assert F[1] == A[self.shape_axes] + self.window_size - 1
        assert F[0] == A[0]

        return F[0], A[self.shape_axes], F[-1]

    def call(self, inputs, mask=None):
        #works only for 2 inputs
        if type(inputs) is not list or not len(inputs) == 2:
            raise Exception('Merge must be called on a list of tensors '
                            '(at least 2). Got: ' + str(inputs))
        #shape F = N, s + w - 1, 1, nkernels
        F = inputs[0]
        #shape A: N, s0, s1
        A = inputs[1]
        #shape a = s (depends on transform axis)
        a = K.sum(A, axis=self.transform_axis)
        a = K.expand_dims(a, dim=-1)

        # shape = N, s, nfilter
        a = K.repeat_elements(a, F.shape[-1], axis=-1)

        F_shuffled = K.permute_dimensions(F, pattern=(0,2,1,3))

        #shape = N, 1, s, nfilter
        F_p = pool_2d(F_shuffled, ds=(self.window_size, 1), ignore_border=True, st=(1, 1), mode='average_inc_pad')
        # shape = N, s, nfilter
        F_p = K.reshape(F_p, (F_p.shape[0], F_p.shape[2], F_p.shape[3]))
        # shape = N, s, nfilter
        return F_p * a
