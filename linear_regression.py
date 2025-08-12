import numpy as np


class LinearRegression:
    def __init__(self, x_feature: np.ndarray, y_target: np.ndarray, 
                 stopping_threshold: complex = 1e-6, max_iterations: int = 1000, learning_rate: float = 0.0001):
        
        self.x_feature = x_feature
        self.y_target = y_target
        self.stopping_threshold = stopping_threshold
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        
        self.current_weight = 0.1 
        self.current_bias = 0.01
        self.previous_cost = None
        self.current_cost = None
        self.n = len(x_feature)

        self.costs = []
        self.weights = []
    
    def predict(self):
        return self.current_weight * self.x_feature + self.current_bias

    def mean_squard_error(self, y_preticted):
        return np.mean(np.sum((y_preticted - self.y_target) ** 2))

    def gradient_descent(self):
        
        for i in range(self.max_iterations):
            y_predicted = self.predict()
            self.current_cost = self.mean_squard_error(y_predicted)

            # Logging VOR dem Konvergenz-Check
            self.costs.append(self.current_cost)
            self.weights.append(self.current_weight)

            if self.previous_cost is not None and abs(self.current_cost - self.previous_cost) <= self.stopping_threshold:
                break

            self.previous_cost = self.current_cost

            weight_derivative = (1 / self.n) * np.sum((y_predicted - self.y_target) * self.x_feature)
            bias_derivative = (1 / self.n) * np.sum(y_predicted - self.y_target)

            self.current_weight -= self.learning_rate * weight_derivative
            self.current_bias -= self.learning_rate * bias_derivative

            
        return self.current_weight, self.current_bias
    
    
    