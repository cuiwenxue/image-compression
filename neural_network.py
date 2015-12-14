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

    def __init__(self, input_layer_size=64, hidden_layer_size=16, output_layer_size=64, learning_rate=0.5):
        self.learning_rate = learning_rate
        self.input_layer = [Node() for i in range(input_layer_size)]
        self.v_edges = [[0 for i in range(hidden_layer_size)] for i in range(input_layer_size)]
        self.hidden_layer = [Node() for i in range(hidden_layer_size)]
        self.w_edges = [[0 for i in range(output_layer_size)] for i in range(hidden_layer_size)]
        self.output_layer = [Node() for i in range(output_layer_size)]

    def get_output(self, input):
        if len(input) != len(self.input_layer):
            raise NeuralNetworkException('Improper input size')

        for i in range(len(self.hidden_layer)):
            self.hidden_layer[i].value = self._calculate_output(i, hidden_layer=True)
        for i in range(len(self.output_layer)):
            self.output_layer[i].value = self._calculate_output(i, hidden_layer=False)
        return [self.output_layer[i].value for i in range(len(self.output_layer))]

    def teach_step(self, input, target):
        if len(input) != len(self.input_layer):
            raise NeuralNetworkException('Improper input size')
        if len(target) != len(self.output_layer):
            raise NeuralNetworkException('Improper target size')

        for i in range(len(self.hidden_layer)):
            self.hidden_layer[i].value = self._calculate_output(i, hidden_layer=True)
        for i in range(len(self.output_layer)):
            self.output_layer[i].value = self._calculate_output(i, hidden_layer=False)

        for i in range(len(self.output_layer)):
            self.output_layer[i].error = self._calculate_error(i, target[i], hidden_layer=False)
        for i in range(len(self.hidden_layer)):
            for j in range(len(self.output_layer)):
                self.w_edges[i][j] = self._update_weight(i, j, self.w_edges)

        for i in range(len(self.hidden_layer)):
            self.hidden_layer[i].error = self._calculate_error(i, hidden_layer=True)
        for i in range(len(self.input_layer)):
            for j in range(len(self.hidden_layer)):
                self.v_edges[i][j] = self._update_weight(i, j, self.v_edges)

    def init_weights(self):
        for i in range(len(self.input_layer)):
            for j in range(len(self.hidden_layer)):
                self.v_edges[i][j] = random.uniform(-1, 1)

        for i in range(len(self.hidden_layer)):
            for j in range(len(self.output_layer)):
                self.w_edges[i][j] = random.uniform(-1, 1)

    def _update_weight(self, i, j, edges):
        if self.v_edges == edges:
            return self.v_edges[i][j] + self.learning_rate * self.hidden_layer[j].error * self.input_layer[i].value
        else:
            return self.w_edges[i][j] + self.learning_rate * self.output_layer[j].error * self.hidden_layer[i].value

    def _calculate_error(self, node_index, target=None, hidden_layer=True):
        if hidden_layer:
            weighted_error_sum = 0
            for i in range(len(self.output_layer)):
                weighted_error_sum += self.w_edges[node_index][i] * self.output_layer[i].error
            return weighted_error_sum * self.hidden_layer[node_index].value * (1 - self.hidden_layer[node_index].value)
        else:
            return self.output_layer[node_index].value * (1 - self.output_layer[node_index].value) * (target - self.output_layer[node_index].value)

    def _calculate_output(self, node_index, hidden_layer=True):
        result = 0
        if hidden_layer:
            for i in range(len(self.input_layer)):
                result += self.v_edges[i][node_index] * self.input_layer[i].value
            return sigmoid_function(result)
        else:
            for i in range(len(self.hidden_layer)):
                result += self.w_edges[i][node_index] * self.hidden_layer[i].value
            return sigmoid_function(result)


class Node(object):
    """Class represents neural network node
    """

    def __init__(self, value=0, error=0):
        self.value = value
        self.error = error