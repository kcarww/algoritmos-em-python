from typing import List
from math import exp

def dot_product(xs: List[float], ys: List[float]) -> float:
    return sum(x * y for x, y in zip(xs, ys))

def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))

def derivative_sigmoid(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))

