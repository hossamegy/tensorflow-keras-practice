import numpy as np

class FullyConnectedLayer:
    def __init__(self, units, activation):
        self.units = units
        self.activation = activation
        self.w = None
        self.b = None

    def build(self, input_shape):
        self.w = np.random.randn(self.units, input_shape) * 0.01
        self.b = np.zeros((self.units, 1))

    def forward(self, input):
        if self.w is None or self.b is None:
            self.build(input.shape[0])

        self.input = input
        self.z = np.dot(self.w, input) + self.b
        self.a = self.activate(self.z)
        return self.a

    def activate(self, z):
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'linear':
            return z

    def activation_derivative(self):
        if self.activation == 'relu':
            return np.where(self.z > 0, 1, 0)
        elif self.activation == 'sigmoid':
            sig = self.activate(self.z)
            return sig * (1 - sig)
        elif self.activation == 'linear':
            return np.ones_like(self.z)

class SequentialModel:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input):
        self.inputs = [input]
        for layer in self.layers:
            input = layer.forward(input)
            self.inputs.append(input)
        return self.inputs[-1]

    def backward(self, loss_grad, learning_rate=0.01):
        grad = loss_grad
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            activation_deriv = layer.activation_derivative()
            grad = grad * activation_deriv

            input_prev = self.inputs[i]
            grad_w = np.dot(grad, input_prev.T)
            grad_b = np.sum(grad, axis=1, keepdims=True)
            grad_input = np.dot(layer.w.T, grad)

            layer.w -= learning_rate * grad_w
            layer.b -= learning_rate * grad_b
            grad = grad_input

    def compute_loss_grad(self, predicted, actual):
        return 2 * (predicted - actual) / actual.size

    def compute_loss(self, predicted, actual):
        return np.mean((predicted - actual) ** 2)

    def train(self, input_data, output_data, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            predicted = self.forward(input_data)
            loss = self.compute_loss(predicted, output_data)
            loss_grad = self.compute_loss_grad(predicted, output_data)
            self.backward(loss_grad, learning_rate)
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}')


    def predict(self, input_data):
        return self.forward(input_data)

# generate test data
def generate_data(n_samples):
    x = np.arange(n_samples).reshape(1, -1)
    y = (8 * x - 1).reshape(1, -1)
    return x, y

input_data, output_data = generate_data(100)

# test model
model1 = SequentialModel([
    FullyConnectedLayer(15, 'relu'),
    FullyConnectedLayer(1, 'linear')
])
# training
model1.train(input_data, output_data, epochs=1000000, learning_rate=0.0001)
