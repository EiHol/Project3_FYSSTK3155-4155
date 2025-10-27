"""
This file contains an implementation of a feed forward neural network
"""

import autograd.numpy as np
from autograd import grad

class NeuralNetwork:
    def __init__(
        self,
        network_input_size,
        layer_output_sizes,
        activation_funcs,
        cost_func,
    ):
        self.network_input_size = network_input_size
        self.layer_output_sizes = layer_output_sizes
        self.activation_funcs = activation_funcs
        self.cost_func = cost_func
        self.layers = self._create_layers()
        self.gradient_func = self._create_gradient_func()
        self.velocities = None

    
    def _create_layers(self):
        """Creates layers for the neural network based on initial network input size and layer output sizes"""

        # Empty list to store layers
        layers = []

        i_size = self.network_input_size

        # Iterate through layer output sizes and append necessary weights and biases to that layer
        for layer_output_size in self.layer_output_sizes:
            # Initialize weight and bias values to random values
            W = np.random.randn(i_size, layer_output_size)
            b = np.random.randn(1, layer_output_size)
            
            # Append weights and biases to the layer
            layers.append((W, b))

            # Update input size for next layer
            i_size = layer_output_size
            
        # Return the list of weights and biases for each layer
        return layers


    def predict(self, inputs):
        """Performs forward propagation through the network to compute predictions"""

        # Set a equal to the initial input value
        a = inputs

        # Iterate through layers and their respective activation functions to compute the prediction a
        for (W, b), activation_func in zip(self.layers, self.activation_funcs):
            # Compute weighted sum
            z = a @ W + b
            # Apply activation function
            a = activation_func(z)

        # Return the final prediction
        return a


    def _create_gradient_func(self):
        """Creates gradient function using autograd for automatic differentiation"""
        from autograd import grad
        
        def cost(layers, inputs, targets):
            # Forward pass through network
            a = inputs
            for (W, b), activation_func in zip(layers, self.activation_funcs):
                z = a @ W + b
                a = activation_func(z)
            # Return cost function value
            return self.cost_func(a, targets)
        
        # Create gradient function with respect to layers
        gradient_func = grad(cost, 0)
        return gradient_func


    def compute_gradients(self, inputs, targets):
        """Computes the gradients for each layer and returns all gradients"""
        layers_grad = self.gradient_func(self.layers, inputs, targets)
        return layers_grad


    def update_weights(self, layers_grad, eta=0.001):
        """Standard gradient descent for updating weights and biases"""
        # Iterate through layers and their gradients
        for j, ((W, b), (W_g, b_g)) in enumerate(zip(self.layers, layers_grad)):
            # Update weights and biases using gradient descent
            W -= W_g * eta
            b -= b_g * eta
            # Store updated layer
            self.layers[j] = (W, b)


    def update_weights_momentum(self, layers_grad, eta=0.001, alpha=0.9):
        """Gradient descent with momentum for updating weights and biases"""
        # Initialize velocities to 0 if velocities dont exist yet
        if self.velocities is None:
            # List to store velocities
            self.velocities = []
            for W, b in self.layers:
                # Set weight velocities to 0
                W_v = np.zeros_like(W)
                # Set bias velocities to 0
                b_v = np.zeros_like(b)
                # Append velicities to the list
                self.velocities.append((W_v, b_v))

        # For each weight and bias, and their corresponding gradient and velocity
        for indx, ((W, b), (W_g, b_g), (W_v, b_v)) in enumerate(zip(self.layers, layers_grad, self.velocities)):

            # Compute the velocity of the weight and bias
            W_v = alpha * W_v - eta * W_g
            b_v = alpha * b_v - eta * b_g

            # Compute the new weight and bias values
            W_n = W + W_v
            b_n = b + b_v
            
            # Update weight, bias, and velocity of the current layer
            self.layers[indx] = (W_n, b_n)
            self.velocities[indx] = (W_v, b_v)



def train_network(neural_network, inputs, targets, eta=0.01, epochs=100):
    """Trains the neural network using standard gradient descent"""
    # Iterate through epochs
    for i in range(epochs):
        # Compute gradients for all layers
        layers_grad = neural_network.compute_gradients(inputs, targets)
        # Update weights using gradient descent
        neural_network.update_weights(layers_grad, eta)
    
    # Return final trained layers
    return neural_network.layers


def train_network_momentum(neural_network, inputs, targets, eta=0.01, alpha=0.9, epochs=100):
    """Trains the neural network using gradient descent with momentum"""
    # Iterate through epochs
    for i in range(epochs):
        # Compute gradients for all layers
        layers_grad = neural_network.compute_gradients(inputs, targets)
        # Update weights using momentum
        neural_network.update_weights_momentum(layers_grad, eta, alpha)
    
    # Return final trained layers
    return neural_network.layers


def train_network_stochastic_momentum(neural_network, inputs, targets, eta=0.01, alpha=0.9, epochs=100, batch_size=25):
    """Trains the neural network using stochastic gradient descent with momentum"""
    # Iterate through epochs
    for i in range(epochs):
        # Randomly shuffle the data
        p = np.random.permutation(len(inputs))
        shuffled_inputs = inputs[p]
        shuffled_targets = targets[p]

        # Iterate through mini-batches
        for j in range(0, len(inputs), batch_size):
            # Extract current batch
            batch_inputs = shuffled_inputs[j : j + batch_size]
            batch_targets = shuffled_targets[j : j + batch_size]

            # Compute gradients for the batch
            layers_grad = neural_network.compute_gradients(batch_inputs, batch_targets)
            # Update weights using stochastic momentum
            neural_network.update_weights_momentum(layers_grad, eta, alpha)
    
    # Return final trained layers
    return neural_network.layers