from typing import List
from math import exp

def dot_product(xs: List[float], ys: List[float]) -> float:
    return sum(x * y for x, y in zip(xs, ys))

def sigmoid(x: float) -> float:
    return 1 / (1 + exp(-x))

def derivative_sigmoid(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))

def normalize_by_feature_scaling(dataset: List[List[float]]) -> List[List[float]]:
    for col_num in range(len(dataset[0])):
        column: List[float] = [row[col_num] for row in dataset]
        maximum = max(column)
        minimum = min(column)
        for row_num in range(len(dataset)):
            dataset[row_num][col_num] = (dataset[row_num][col_num] - minimum) / (maximum - minimum)