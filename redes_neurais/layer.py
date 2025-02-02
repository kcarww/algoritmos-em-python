from typing import List, Optional, Callable
from random import random
from neuron import Neuron
from util import dot_product

class Layer:
    def __init__(self, previous_layer: Optional['Layer'], num_neurons: int, learning_rate: float, activation_function: Callable[[float], float], derivative_activation_function: Callable[[float], float]) -> None:
        self.neurons: List[Neuron] = []
        self.previous_layer: Optional['Layer'] = previous_layer
        
        for _ in range(num_neurons):
            if previous_layer is None:
                random_weights: List[float] = []
            else:
                random_weights = [random() for _ in range(len(previous_layer.neurons))]
            
            neuron: Neuron = Neuron(random_weights, learning_rate, activation_function, derivative_activation_function)
            self.neurons.append(neuron)
        self.output_cache: List[float] = [0.0 for _ in range(num_neurons)]

    def outputs(self, inputs: List[float]) -> List[float]:
        if self.previous_layer is None:
            self.output_cache = inputs
        else:
            self.output_cache = [neuron.output(inputs) for neuron in self.neurons]
        return self.output_cache