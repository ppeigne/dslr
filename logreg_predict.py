import pandas as pd
import numpy as np
import argparse
from joblib import load

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from logistic import LogisticRegression

def load_one_vs_all_logistic_reg():
	return [LogisticRegression().load_params(f"p{i}") for i in range(4)]

def one_vs_all_predict(X, models):
	# Collect predictions for each model
	predictions = np.concatenate([model.predict(X) for model in models], axis=1)
	# Select higher prediction class for each prediction
	return np.argmax(predictions, axis=1)

def parser():
    my_parser = argparse.ArgumentParser(description='Train a logistic regression model.')

    my_parser.add_argument('dataset', help='the dataset you want to train on.')

    return my_parser.parse_args()


if __name__ == '__main__':
	args = parser()
	data_path = args.dataset

	if data_path != 'datasets/dataset_test.csv':
		print('Error: wrong dataset given. Please give the proper dataset to test on.')
	
	else:
		try:
			# Load data
			x_test = pd.read_csv(data_path, index_col='Index')[["Defense Against the Dark Arts", "Herbology", "Charms", "Ancient Runes"]]

			# Fill missing values
			imputer = SimpleImputer()
			x_test = x_test
			x_test = imputer.fit_transform(x_test)

			# Normalize data
			scaler = load("scaler.joblib")
			x_test = scaler.transform(x_test)

			# Generate one vs all models
			models = load_one_vs_all_logistic_reg()

			# Predict
			prev = pd.DataFrame(one_vs_all_predict(x_test, models), columns=['Hogwarts House'])
			prev.index.name = 'Index'
			h_map = {0: "Ravenclaw", 1: "Slytherin", 2: "Gryffindor", 3: "Hufflepuff"}
			prev['Hogwarts House'] = prev['Hogwarts House'].map(h_map)

			# Save prediction
			prev.to_csv('houses.csv', header=True, index=True)
		except:
			print('Something wrong happende. Did you train your model before trying to predict with it?')