"""In this module you can find implementation of neural network.
   NeuralNetwork allows to create network with input and output layer and 0 or more hidden layers.
   All layer must contain at least single neuron.
"""
import math
import random
import pickle


def sigmoid_function(x):
    return 1 / (1 + math.exp(-x))


def save(neural_network, filename):
    if type(neural_network) is NeuralNetwork:
        pickle.dump(neural_network, open(filename + '.zdp', 'wb'), pickle.HIGHEST_PROTOCOL)
    else:
        raise NeuralNetworkException('Given neural network is not a type of ' + NeuralNetwork.__name__)


def load(filename):
    neural_network = pickle.load(open(filename, 'rb'))
    if type(neural_network) is not NeuralNetwork:
        raise NeuralNetworkException('Loaded neural network is not a type of ' + NeuralNetwork.__name__)
    return neural_network


class NeuralNetworkException(Exception):
    pass


class NeuralNetwork(object):
    """Neural network containing single hidden layer.
       Teaching algorithm uses error backpropagation.
    """

    def __init__(self, input_layer_size, hidden_layers_sizes, output_layer_size, learning_rate=0.5):
        self.learning_rate = learning_rate
        self.input_layer = Layer(input_layer_size)
        self.hidden_layers = [Layer(hidden_layers_sizes[i]) for i in xrange(len(hidden_layers_sizes))]
        self.output_layer = Layer(output_layer_size)
        self._init_edges(self.input_layer)
        for hidden_layer in self.hidden_layers:
            self._init_edges(hidden_layer)

    def init_weights(self):
        """Initialize all network edges with random values from <-1; 1>
        """
        for neuron in self.input_layer:
            for edge in neuron.outgoing_edges:
                edge.weight = random.uniform(-1, 1)

        for hidden_layer in self.hidden_layers:
            for neuron in hidden_layer:
                for edge in neuron.outgoing_edges:
                    edge.weight = random.uniform(-1, 1)

    def run(self, input):
        if len(input) != len(self.input_layer):
            raise NeuralNetworkException('Improper input size')

        for i, neuron in enumerate(self.input_layer, start=0):
            neuron.value = input[i]

        for hidden_layer in self.hidden_layers:
            hidden_layer.update_values()
        self.output_layer.update_values()

        return [neuron.value for neuron in self.output_layer]

    def teach_step(self, input, target):
        """Calculate network output. Compare it with target and calculate errors.
           Propagate errors to predecessing layers and adjusts edges weights.
        """
        if len(input) != len(self.input_layer):
            raise NeuralNetworkException('Improper input size')
        if len(target) != len(self.output_layer):
            raise NeuralNetworkException('Improper target size')

        self.run(input)
        self._propagate_error(target)

    def _init_edges(self, layer):
        """Creates all possible connection between neurons in network
        """
        for neuron_begin in layer:
            for neuron_end in self._get_layer_successor(layer):
                edge = Edge(neuron_begin, neuron_end)
                neuron_begin.outgoing_edges.append(edge)
                neuron_end.ingoing_edges.append(edge)

    def _propagate_error(self, target):
        for i, neuron in enumerate(self.output_layer, start=0):
            neuron.update_error(target[i])
            for edge in neuron.ingoing_edges:
                edge.update_weight(self.learning_rate)

        for hidden_layer in self.hidden_layers:
            for neuron in hidden_layer:
                neuron.update_error()
                for edge in neuron.ingoing_edges:
                    edge.update_weight(self.learning_rate)

    def _get_layer_successor(self, layer):
        if layer is self.input_layer:
            if len(self.hidden_layers) > 0:
                return self.hidden_layers[0]
            else:
                return self.output_layer
        elif layer in self.hidden_layers:
            if self.hidden_layers.index(layer) == len(self.hidden_layers) - 1:
                return self.output_layer
            else:
                return self.hidden_layers[self.hidden_layers.index(layer) + 1]
        elif layer is self.output_layer:
            raise NeuralNetworkException('Output layer has no successor layer')
        else:
            raise NeuralNetworkException('Given layer unknown')


class Layer(object):
    """Class represents neural network layer
    """

    def __init__(self, size):
        self.neurons = [Neuron() for i in xrange(size)]

    def __getitem__(self, index):
        return self.neurons[index]

    def __len__(self):
        return len(self.neurons)

    def update_values(self):
        for neuron in self.neurons:
            neuron.update_value()


class Edge(object):
    """Class represents neural network edge connecting two neurons
    """

    def __init__(self, begin=None, end=None, weight=0.0):
        self.begin = begin
        self.end = end
        self.weight = weight

    def update_weight(self, learning_rate):
        self.weight += learning_rate * self.begin.value * self.end.error


class Neuron(object):
    """Class represents neural network node
    """

    def __init__(self, value=0.0, error=0.0):
        self.value = value
        self.error = error
        self.ingoing_edges = []
        self.outgoing_edges = []

    def update_value(self):
        weighted_value_sum = 0.0
        for edge in self.ingoing_edges:
            weighted_value_sum += edge.weight * edge.begin.value
        self.value = sigmoid_function(weighted_value_sum)

    def update_error(self, target=None):
        if target is not None:
            self.error = self.value * (1 - self.value) * (target - self.value)
        else:
            weighted_error_sum = 0.0
            for edge in self.outgoing_edges:
                weighted_error_sum += edge.weight * edge.end.error
            self.error = weighted_error_sum * self.value * (1 - self.value)