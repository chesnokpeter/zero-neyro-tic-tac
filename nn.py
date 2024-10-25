import random
import math
import json

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class Neuron:
    def __init__(self, num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)
        self.output = 0
        self.inputs = []

    def activate(self, inputs):
        self.inputs = inputs
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        self.output = sigmoid(weighted_sum)
        return self.output

    def update_weights(self, learning_rate, delta):
        for i in range(len(self.weights)):
            self.weights[i] += learning_rate * delta * self.inputs[i]
        self.bias += learning_rate * delta

class Layer:
    def __init__(self, num_neurons, num_inputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]

    def forward(self, inputs):
        outputs = [neuron.activate(inputs) for neuron in self.neurons]
        return outputs

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = Layer(hidden_size, input_size)
        self.output_layer = Layer(output_size, hidden_size)

    def forward(self, inputs):
        hidden_output = self.hidden_layer.forward(inputs)
        output = self.output_layer.forward(hidden_output)
        return output

    def backward(self, inputs, expected_output, actual_output, learning_rate):
        output_errors = [(expected - actual) for expected, actual in zip(expected_output, actual_output)]

        output_deltas = [error * sigmoid_derivative(neuron.output) for neuron, error in zip(self.output_layer.neurons, output_errors)]

        hidden_errors = [0] * len(self.hidden_layer.neurons)
        for i, hidden_neuron in enumerate(self.hidden_layer.neurons):
            hidden_errors[i] = sum(output_deltas[j] * output_neuron.weights[i] for j, output_neuron in enumerate(self.output_layer.neurons))

        hidden_deltas = [error * sigmoid_derivative(neuron.output) for neuron, error in zip(self.hidden_layer.neurons, hidden_errors)]

        hidden_outputs = [neuron.output for neuron in self.hidden_layer.neurons]
        for i, neuron in enumerate(self.output_layer.neurons):
            neuron.update_weights(learning_rate, output_deltas[i])

        for i, neuron in enumerate(self.hidden_layer.neurons):
            neuron.update_weights(learning_rate, hidden_deltas[i])

    def train(self, training_data, epochs, learning_rate):
        for epoch in range(epochs):
            total_error = 0
            for inputs, expected_output in training_data:
                actual_output = self.forward(inputs)
                total_error += sum((expected - actual) ** 2 for expected, actual in zip(expected_output, actual_output))
                self.backward(inputs, expected_output, actual_output, learning_rate)
            print(f'Epoch {epoch + 1}/{epochs}, Error: {total_error}')




if __name__ == "__main__":
    nn = NeuralNetwork(input_size=9, hidden_size=9, output_size=9)

    # training_data = [
    #     ([0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0]),  # Пустое поле, ожидаем ход в 1-ю клетку
    #     ([1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0]),  # После хода крестика в 1-ю клетку — ход во 2-ю
    #     ([1, 2, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]),  # Ходим в 3-ю клетку после хода крестика и нолика
    # ]

    dataser = open("dataset.txt", "r")

    training_data = []

    while True:
        line = dataser.readline()
        if not line:
            break
        training_data.append(json.loads(line))
        print(json.loads(line))

    dataser.close()

    # print(training_data)


    nn.train(training_data, epochs=1000, learning_rate=0.1)

    test_input = [2, 1, 0, 2, 1, 0, 0, 0, 0]  # Текущая ситуация на игровом поле (крестик в 1-й клетке)
    output = nn.forward(test_input)

    print("Предсказание сети:", output)

    predicted_move = output.index(max(output))
    print(f"Нейросеть рекомендует поставить в клетку {predicted_move + 1}")


