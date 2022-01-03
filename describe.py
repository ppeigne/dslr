import argparse
import numpy as np
import pandas as pd

def count(data):
	return np.logical_not(np.isnan(data)).sum()

def stats(data):
	count_ = count(data)
	if count_ == 0:
		mean_ = 0
		std = 0
	else:
		mean_ = np.nansum(data) / count_
		std = np.sqrt(np.nansum((data - mean_) ** 2) / count_)
	stats = {
		'count': count_,
		'mean': mean_,
		'std': std
	}
	stats['var'] = std ** 2
	return stats

def quartile(data):
	def get_quartile(data, q):
		count_ = count(sorted_data) - 1
		q_ = q * count_

		if q_.is_integer():
			return sorted_data[int(q_)]
		else:
			q_ = int(q_)
			return (sorted_data[q_] + sorted_data[q_ + 1]) / 2

	sorted_data = np.sort(data)
	sorted_data = sorted_data[~np.isnan(sorted_data)]
	if len(sorted_data) == 0:
		quartiles = {
					'min' : 0, 
					25 : 0,
					50 : 0,
					75 : 0,
					'max' : 0 	
					}
		return quartiles

	quartiles = {
		'min': sorted_data[0],
		25: get_quartile(sorted_data, .25),
		50: get_quartile(sorted_data, .50),
		75: get_quartile(sorted_data, .75),
		'max': sorted_data[-1],
	}
	quartiles['IQR'] = quartiles[75] - quartiles[25]
	return quartiles

def describe(data):
	num_features = [c for c in data.columns if data[c].dtype == float]
	res = pd.DataFrame([{**stats(data[c].to_numpy()), **quartile(data[c].to_numpy())} for c in num_features]).T
	res.columns = num_features
	return res.round(3)


def parser():
	my_parser = argparse.ArgumentParser(description='Train a logistic regression model.')

	my_parser.add_argument('dataset', help='the dataset you want to train on.')

	return my_parser.parse_args()


if __name__ == '__main__':
	args = parser()
	data_path = args.dataset

	if data_path == None:
		print('Error: no dataset given. Please give the proper dataset to test on.')
	else:	
		try:
			data = pd.read_csv(data_path).dropna(axis=1, how='all')
			print(describe(data))
		except:
			print('There is something wrong with the dataset.')