import pandas as pd
import numpy as np
import argparse
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from logistic import LogisticRegression

def generate_one_vs_all_logistic_reg(X, y):
	classes = np.unique(y)		
	return [LogisticRegression().fit(X, y == c) for c in classes]

def one_vs_all_save(models):
	for i, x in enumerate(models):
		x.save_params(name=f"p{i}")

def one_vs_all_predict(X, models):
	# Collect predictions for each model
	predictions = np.concatenate([model.predict(X) for model in models], axis=1)
	# Select higher prediction class for each prediction
	return np.argmax(predictions, axis=1)


def parser():
    my_parser = argparse.ArgumentParser(description='Train a logistic regression model.')

    my_parser.add_argument('dataset',
                        help='the dataset you want to train on.')

    return my_parser.parse_args()

if __name__ == '__main__':
	args = parser()
	data_path = args.dataset

	if data_path != 'datasets/dataset_train.csv':
		print('Error: wrong dataset given. Please give the proper dataset to train on.')
	
	else:

		# Load data
		x_train = pd.read_csv(data_path, index_col='Index')[["Hogwarts House","Defense Against the Dark Arts", "Herbology", "Charms", "Ancient Runes"]]
		# x_train = pd.read_csv('datasets/dataset_train.csv', index_col='Index')[["Hogwarts House","Defense Against the Dark Arts", "Herbology", "Charms", "Ancient Runes"]]

		# Convert Hogwarts House as number
		h_map = {"Ravenclaw": 0, "Slytherin": 1, "Gryffindor": 2, "Hufflepuff": 3}
		x_train['Hogwarts House'] = x_train['Hogwarts House'].map(h_map)
		
		y_train = np.array(x_train['Hogwarts House'])

		# Fill missing values
		imputer = SimpleImputer()
		x_train = x_train[["Defense Against the Dark Arts", "Herbology", "Charms", "Ancient Runes"]]
		x_train = imputer.fit_transform(x_train)

		x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, random_state=42)
		
		# Normalize data
		scaler = RobustScaler()
		x_train = scaler.fit_transform(x_train)
		x_test = scaler.transform(x_test)

		# Generate one vs all models
		models = generate_one_vs_all_logistic_reg(x_train, y_train)

		# Save models
		dump(scaler, "scaler.joblib")
		one_vs_all_save(models)

		prev = one_vs_all_predict(x_test, models)

		# M2JS3SKS7SV


		# p1 = LogisticRegression()
		# p1.fit(x_train, y_train == 0)
		# p1.save_params(name='p1.json')
		# prev1 = p1.predict(x_test)
		# print(f"Prev1 score = {accuracy_score(y_test==0, prev1 > 0)}")

		# p2 = LogisticRegression()
		# p2.fit(x_train, y_train == 1)
		# p2.save_params(name='p2.json')

		# prev2 = p2.predict(x_test)
		# print(f"Prev2 score = {accuracy_score(y_test==1, prev2 > 0)}")

		# p3 = LogisticRegression()
		# p3.fit(x_train, y_train == 2)
		# p3.save_params(name='p3.json')

		# prev3 = p3.predict(x_test)
		# print(f"Prev3 score = {accuracy_score(y_test==2, prev3 > 0)}")

		# p4 = LogisticRegression()
		# p4.fit(x_train, y_train == 3)
		# p4.save_params(name='p4.json')

		# prev4 = p4.predict(x_test)
		# print(f"Prev4 score = {accuracy_score(y_test==3, prev4 > 0)}")


		# print(np.concatenate((prev1, prev2, prev3, prev4), axis=1))

		# prev = pd.DataFrame(np.concatenate((prev1, prev2, prev3, prev4), axis=1)).idxmax(axis=1)
		# print(pd.concat((prev, pd.DataFrame(y_test)), axis=1))

		met = accuracy_score(y_test, prev)
		print(met)