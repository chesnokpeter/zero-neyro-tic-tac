import random
import math
import json
import pickle

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

class Neuron:
    def __init__(self, num_inputs, activation_function, activation_derivative):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.output = 0
        self.inputs = []

    def activate(self, inputs):
        self.inputs = inputs
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        self.output = self.activation_function(weighted_sum)
        return self.output

    def update_weights(self, learning_rate, delta):
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * delta * self.inputs[i]
        self.bias += learning_rate * delta

class Layer:
    def __init__(self, num_neurons, num_inputs, activation_function, activation_derivative):
        self.neurons = [Neuron(num_inputs, activation_function, activation_derivative) for _ in range(num_neurons)]

    def forward(self, inputs):
        outputs = [neuron.activate(inputs) for neuron in self.neurons]
        return outputs

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, second_hidden_size, output_size):
        self.hidden_layer = Layer(hidden_size, input_size, relu, relu_derivative)
        self.second_hidden_layer = Layer(second_hidden_size, hidden_size, relu, relu_derivative)
        self.output_layer = Layer(output_size, second_hidden_size, sigmoid, sigmoid_derivative)

    def forward(self, inputs):
        hidden_output = self.hidden_layer.forward(inputs)
        second_hidden_output = self.second_hidden_layer.forward(hidden_output)
        output = self.output_layer.forward(second_hidden_output)
        return output

    def backward(self, inputs, expected_output, actual_output, learning_rate):
        output_errors = [(expected - actual) for expected, actual in zip(expected_output, actual_output)]

        output_deltas = [error * sigmoid_derivative(neuron.output) for neuron, error in zip(self.output_layer.neurons, output_errors)]

        hidden_errors = [0] * len(self.second_hidden_layer.neurons)
        for i, hidden_neuron in enumerate(self.second_hidden_layer.neurons):
            hidden_errors[i] = sum(output_deltas[j] * output_neuron.weights[i] for j, output_neuron in enumerate(self.output_layer.neurons))

        hidden_deltas = [error * relu_derivative(neuron.output) for neuron, error in zip(self.second_hidden_layer.neurons, hidden_errors)]

        first_hidden_errors = [0] * len(self.hidden_layer.neurons)
        for i, hidden_neuron in enumerate(self.hidden_layer.neurons):
            first_hidden_errors[i] = sum(hidden_deltas[j] * second_hidden_neuron.weights[i] for j, second_hidden_neuron in enumerate(self.second_hidden_layer.neurons))

        first_hidden_deltas = [error * relu_derivative(neuron.output) for neuron, error in zip(self.hidden_layer.neurons, first_hidden_errors)]

        second_hidden_outputs = [neuron.output for neuron in self.second_hidden_layer.neurons]
        for i, neuron in enumerate(self.output_layer.neurons):
            neuron.update_weights(learning_rate, output_deltas[i])

        for i, neuron in enumerate(self.second_hidden_layer.neurons):
            neuron.update_weights(learning_rate, hidden_deltas[i])

        for i, neuron in enumerate(self.hidden_layer.neurons):
            neuron.update_weights(learning_rate, first_hidden_deltas[i])

    def train(self, training_data, epochs, learning_rate):
        for epoch in range(epochs):
            total_error = 0
            correct_predictions = 0

            for inputs, expected_output in training_data:
                actual_output = self.forward(inputs)
                
                total_error += sum((expected - actual) ** 2 for expected, actual in zip(expected_output, actual_output))
                
                self.backward(inputs, expected_output, actual_output, learning_rate)

                predicted_index = actual_output.index(max(actual_output))
                expected_index = expected_output.index(max(expected_output))

                if predicted_index == expected_index:
                    correct_predictions += 1
            accuracy = correct_predictions / len(training_data)
            print(f'Epoch {epoch + 1}/{epochs}, Error: {total_error}, Accuracy: {accuracy}')

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)


if __name__ == "__main__":
    nn = NeuralNetwork(input_size=9, hidden_size=18, second_hidden_size=18, output_size=9)

    with open("dataset.txt", "r") as dataser:
        training_data = [json.loads(line) for line in dataser]

    nn.train(training_data, epochs=5000, learning_rate=0.001)

    test_input = [2, 1, 0, 2, 1, 0, 0, 0, 0]
    output = nn.forward(test_input)

    print("Network prediction:", output)

    predicted_move = output.index(max(output))
    print(f"Neural network recommends placing in cell {predicted_move + 1}")

    nn.save('1.pkl')