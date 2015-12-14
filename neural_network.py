import math
import random


def sigmoid_function(x):
    return 1 / (1 + math.exp(-x))


class NeuralNetworkException(Exception):
    pass


class NeuralNetwork():
    """Neural network containing single hidden layer.
       Teaching algorithm uses error backpropagation.
    """

    def __init__(self, input_layer_size=64, hidden_layer_size=16, output_layer_size=64, sigmoid_func=sigmoid_function):
        self.input_layer = [0 for i in range(input_layer_size)]
        self.v_edges = [[0 for i in range(hidden_layer_size)] for i in range(input_layer_size)]
        self.hidden_layer = [0 for i in range(hidden_layer_size)]
        self.w_edges = [[0 for i in range(output_layer_size)] for i in range(hidden_layer_size)]
        self.output_layer = [0 for i in range(output_layer_size)]
        self.learning_rate = 0.5
        self.sigmoid_function = sigmoid_func

    def teach_step(self, input):
        if len(input) != len(self.input_layer):
            raise NeuralNetworkException('Improper input size')

    def init_weights(self):
        for i in range(len(self.input_layer)):
            for j in range(len(self.hidden_layer)):
                self.v_edges[i][j] = random.uniform(-1, 1)

        for i in range(len(self.hidden_layer)):
            for j in range(len(self.output_layer)):
                self.w_edges[i][j] = random.uniform(-1, 1)

    def _calculate_output(self, node_index, hidden_layer=True):
        result = 0
        if hidden_layer:
            for i in range(len(self.input_layer)):
                result += self.v_edges[i][node_index] * self.input_layer[i]
            return self.sigmoid_function(result)
        else:
            for i in range(len(self.hidden_layer)):
                result += self.w_edges[i][node_index] * self.hidden_layer[i]
            return self.sigmoid_function(result)
