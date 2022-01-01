import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from logistic import LogisticRegression


x_train = pd.read_csv('datasets/dataset_train.csv', index_col='Index')[["Hogwarts House","Defense Against the Dark Arts", "Herbology", "Divination",
					"Ancient Runes", "History of Magic"]]

h_map = {"Ravenclaw": 0, "Slytherin": 1, "Gryffindor": 2, "Hufflepuff": 3}
x_train['Hogwarts House'] = x_train['Hogwarts House'].map(h_map)
y_train = np.array(x_train['Hogwarts House'])


imputer = SimpleImputer()
x_train = x_train[["Defense Against the Dark Arts", "Herbology", "Divination",
					"Ancient Runes", "History of Magic"]]
x_train = imputer.fit_transform(x_train)


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=42)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

p1 = LogisticRegression()
p1.fit(x_train, y_train == 0)
prev1 = p1.predict(x_test)
print(f"Prev1 score = {accuracy_score(y_test==0, prev1 > 0)}")

p2 = LogisticRegression()
p2.fit(x_train, y_train == 1)
prev2 = p2.predict(x_test)
print(f"Prev2 score = {accuracy_score(y_test==1, prev2 > 0)}")

p3 = LogisticRegression()
p3.fit(x_train, y_train == 2)
prev3 = p3.predict(x_test)
print(f"Prev3 score = {accuracy_score(y_test==2, prev3 > 0)}")

p4 = LogisticRegression()
p4.fit(x_train, y_train == 3)
prev4 = p4.predict(x_test)
print(f"Prev4 score = {accuracy_score(y_test==3, prev4 > 0)}")


print(np.concatenate((prev1, prev2, prev3, prev4), axis=1))

prev = pd.DataFrame(np.concatenate((prev1, prev2, prev3, prev4), axis=1)).idxmax(axis=1)
print(pd.concat((prev, pd.DataFrame(y_test)), axis=1))

met = accuracy_score(y_test, prev)
print(met)