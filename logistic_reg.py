import numpy as np
from optimizers import Optimizer #, MomentumOptimizer_, RMSOptimizer_, AdamOptimizer_

class LinearRegression_():
    def __init__(self, optimizer="gradient"):
        self.params = None
        self.optimizer = self._select_optimizer(optimizer)

    def _select_optimizer(self, optimizer):
        options = {
            "gradient": Optimizer()#(batch_size=64)
        #    "momentum": MomentumOptimizer_(),
        #    "rms": RMSOptimizer_(),
        #    "adam": AdamOptimizer_()
        }
        return options[optimizer]

    def _generate_params(self, n_input):
        self.params= [{}]
        self.params[0]['theta'] = np.random.random((n_input + 1, 1))

    def _add_intercept(self, x):
        return np.concatenate((np.ones((x.shape[0], 1)), x), axis= 1)

    def predict(self, X):
        if not self.params:
            self._generate_params(X.shape[1])
        X_ = self._add_intercept(X)
        return X_ @ self.params[0]['theta']

    def gradient(self, X, y, y_pred):
        X_ = self._add_intercept(X)
        y = y.reshape(-1, 1)
        gradient = [{}]
        gradient[0]['theta'] = X_.T @ (y_pred - y) / X.shape[1]
        return gradient

    def fit(self, X, y, n_cycles=100000, learning_rate=0.1):
        self.optimizer.optimize(self, X, y, n_cycles, learning_rate)

class RidgeRegression_(LinearRegression_):
    def gradient(self, X, y, y_pred):
        m = X.shape[1]
        X_ = self._add_intercept(X)
        y = y.reshape(-1, 1)
        # tmp_theta is used to get the sum of thetas without the intercept part (theta[0])
        tmp_theta = np.copy(self.params[0]['theta'])
        tmp_theta[0] = 0
        gradient = [{}]
        # print(f"grads shape = {(X_.T @ (y_pred - y)).shape}")
        # print(f"theta shape = {(tmp_theta.T @ tmp_theta).shape}")
        gradient[0]['theta'] = (X_.T @ (y_pred - y) + tmp_theta.T @ tmp_theta) / m 
        return gradient
    

class LogisticRegression_(RidgeRegression_):
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        if not self.params:
            self._generate_params(X.shape[1])
        X_ = self._add_intercept(X)
        return self._sigmoid(X_ @ self.params[0]['theta'])


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score , f1_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pandas as pd

x_train = pd.read_csv('datasets/dataset_train.csv', index_col='Index')[['Hogwarts House', 'Arithmancy', 'Herbology', 'Defense Against the Dark Arts',
       'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
       'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms',
       'Flying']]

h_map = {"Ravenclaw": 0, "Slytherin": 1, "Gryffindor": 2, "Hufflepuff": 3}
x_train['Hogwarts House'] = x_train['Hogwarts House'].map(h_map)
y_train = np.array(x_train['Hogwarts House'])


imputer = SimpleImputer()
x_train = x_train[['Arithmancy', 'Herbology', 'Defense Against the Dark Arts',
       'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
       'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms',
       'Flying']]
x_train = imputer.fit_transform(x_train)



x_train, x_test, y_train, y_test = train_test_split(x_train, y_train)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

p1 = LogisticRegression_()
p1.fit(x_train, y_train == 0)
prev1 = p1.predict(x_test)

p2 = LogisticRegression_()
p2.fit(x_train, y_train == 1)
prev2 = p2.predict(x_test)

p3 = LogisticRegression_()
p3.fit(x_train, y_train == 2)
prev3 = p3.predict(x_test)

p4 = LogisticRegression_()
p4.fit(x_train, y_train == 3)
prev4 = p4.predict(x_test)


print(np.concatenate((prev1, prev2, prev3, prev4), axis=1))


prev = pd.DataFrame(np.concatenate((prev1, prev2, prev3, prev4), axis=1)).idxmax(axis=1)
print(pd.concat((prev, pd.DataFrame(y_test)), axis=1))

met = accuracy_score(y_test, prev)
print(met)