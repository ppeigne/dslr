import pandas as pd
import numpy as np
import argparse
from joblib import dump

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

		# Convert Hogwarts House as number
		h_map = {"Ravenclaw": 0, "Slytherin": 1, "Gryffindor": 2, "Hufflepuff": 3}
		x_train['Hogwarts House'] = x_train['Hogwarts House'].map(h_map)
		
		y_train = np.array(x_train['Hogwarts House'])

		# Fill missing values
		imputer = SimpleImputer()
		x_train = x_train[["Defense Against the Dark Arts", "Herbology", "Charms", "Ancient Runes"]]
		x_train = imputer.fit_transform(x_train)
		
		# Normalize data
		scaler = RobustScaler()
		x_train = scaler.fit_transform(x_train)

		# Generate one vs all models
		models = generate_one_vs_all_logistic_reg(x_train, y_train)

		# Save models
		dump(scaler, "scaler.joblib")
		one_vs_all_save(models)