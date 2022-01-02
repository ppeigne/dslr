import numpy as np
import json
import copy

class LogisticRegression():
    def __init__(self, learning_rate=1e-1, 
                 batch_size=32, optimizer='rms', 
                 beta=None, gamma=None, 
                 regularization=True, lambda_=1e-4) -> None:
        self.epoch = 0
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.regularization = regularization
        self.beta = beta
        self.gamma = gamma
        self.lambda_ = lambda_
        self.params = None
        self.update_params = None


    def _generate_params(self, n_inputs):
        self.params = np.random.rand(n_inputs + 1, 1)

    def _generate_params_momentum(self, n_inputs):
        self.params = np.random.rand(n_inputs + 1, 1)
        self.update_params = np.zeros_like(self.params)
        self.beta = self.beta if self.beta else .9
        self.gamma = self.gamma if self.gamma else 1 - self.beta

    def _generate_params_rms(self, n_inputs):
        self.params = np.random.rand(n_inputs + 1, 1)
        self.update_params = np.zeros_like(self.params)
        self.beta = self.beta if self.beta else .99

    def _generate_params_adam(self, n_inputs):
        self.params = np.random.rand(n_inputs + 1, 1)
        self.update_params = [np.zeros_like(self.params), np.zeros_like(self.params)]
        if type(self.beta) == tuple and (self.beta[0] and self.beta[1]):
            self.beta = self.beta
        else:
            self.beta = (.9, .99)
        self.regularization = False

    def generate_params(self, n_inputs):
        params_generator = {
            'gradient_descent': self._generate_params,
            'momentum': self._generate_params_momentum,
            'rms': self._generate_params_momentum,
            'adam': self._generate_params_adam
        }
        params_generator[self.optimizer](n_inputs)

    def _add_intercept(self, x):
        return np.concatenate((np.ones((x.shape[0], 1)), x), axis= 1)

    def _sigmoid(self, x, epsilon=1e-8):
        return 1 / (1 + np.exp(-np.clip(x, a_min=-100, a_max=None) + epsilon))

    def _vanilla_update(self, gradient):
        return - gradient * self.learning_rate
    
    def _momentum_update(self, gradient):
        self.update_params *= self.beta
        self.update_params += self.gamma * gradient
        return - self.update_params * self.learning_rate
    
    def _rms_update(self, gradient):
        self.update_params *= self.beta
        self.update_params += (1 - self.beta) * gradient**2
        return - (gradient / (np.sqrt(self.update_params) + 1e-8)) * self.learning_rate

    def _adam_update(self, gradient):
        # Momentum part
        self.update_params[0] *= self.beta[0] 
        self.update_params[0] += (1 - self.beta[0]) * gradient
        momentum_corrected = self.update_params[0] / (1 - (self.beta[0] ** self.epoch) + 1e-8)
        # RMS part
        self.update_params[1] *= self.beta[1] 
        self.update_params[1] += (1 - self.beta[1]) * (gradient ** 2)
        rms_corrected = self.update_params[1] / (1 - (self.beta[1] ** self.epoch) + 1e-8)
        return - (momentum_corrected / (np.sqrt(rms_corrected) + 1e-8)) * self.learning_rate

    def update_rule(self, gradient):
        update_rules = {
            'gradient_descent': self._vanilla_update,
            'momentum': self._momentum_update,
            'rms': self._rms_update,
            'adam': self._adam_update
        }
        return update_rules[self.optimizer](gradient)

    def _batch_optimization(self, X, y, n_iter):
        self.generate_params(X.shape[1])
        for _ in range(n_iter):
            results = self.predict(X)
            gradient = self.gradient(X, y, results)
            self.params += self.update_rule(self.learning_rate * gradient)
        return self
    
    def _minibatch_batch_optimization(self, X, y, n_iter):
        m, n = X.shape
        self.generate_params(n)
        for self.epoch in range(n_iter):
            for t in range(m // self.batch_size):
                mini_batch_X = X[self.batch_size * t:self.batch_size * (t + 1), :]
                mini_batch_y = y[self.batch_size * t:self.batch_size * (t + 1)]
                results = self.predict(mini_batch_X)
                gradient = self.gradient(mini_batch_X, mini_batch_y, results)
                self.params += self.update_rule(self.learning_rate * gradient)
            rest = m % self.batch_size
            if rest != 0:
                final_batch_X = X[-rest:,:]
                final_batch_y = y[-rest:]
                results = self.predict(final_batch_X)
                gradient = self.gradient(final_batch_X, final_batch_y, results)
                self.params += self.update_rule(self.learning_rate * gradient)
        return self

    def fit(self, X, y, n_iter=1000):
        if self.batch_size == -1:
            return self._batch_optimization(X, y, n_iter)
        else:
            return self._minibatch_batch_optimization(X, y, n_iter)
        
    def predict(self, X):
        if self.params.all() == None:
            self.params = self.generate_params(X.shape[1])
        X_ = self._add_intercept(X)
        return self._sigmoid(X_ @ self.params)

    def gradient(self, X, y, y_pred):
        def vanilla(X_, y, y_pred):
            return (X_.T @ (y_pred - y)) / X_.shape[0]
        
        def ridge(X_, y, y_pred, lambda_, theta):
            return (X_.T @ (y_pred - y) + lambda_ * theta[1:].T @ theta[1:]) / X_.shape[0]

        X_ = self._add_intercept(X)
        y = y.reshape(-1, 1)
        if self.regularization:
            return ridge(X_, y, y_pred, self.lambda_, self.params)
        else: 
            return vanilla(X_, y, y_pred)

    def save_params(self, name="logreg_params", path=""):
        with open(f"{path}{name}.json", 'w') as output:
            json.dump(self, output, default=lambda o: o.__dict__ if type(o) is not np.ndarray else o.tolist(), 
                       sort_keys=True, indent=4)

    def load_params(self, name="logreg_params", path=""):
        with open(f"{path}{name}.json", 'w') as json_file:
            data = json.load(json_file)
            self.batch_size = data['batch_size']
            self.beta = data['beta']
            self.gamma = data['gamma']
            self.regularization = data['regularization']
            self.lambda_ = data['lambda_']
            self.learning_rate = data['learning_rate']
            self.optimizer = data['optimizer']
            self.params = np.array(data['params'])
            self.update_params = np.array(data['batch_size'])


