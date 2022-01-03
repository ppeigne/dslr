import numpy as np
import pandas as pd

def count(data):
    return np.logical_not(np.isnan(data)).sum()

def stats(data):
    count_ = count(data)
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
    return res.T.round(3)



# def mean(data):
#     total = count(data)
#     return np.nansum(data) / total

# def std(data):
#     total = count(data)
#     mean_ = mean(data)
#     return np.sqrt(np.nansum((data - mean_) ** 2) / total)

# def min_(data):
#     min_ = float('inf')
#     for x in data:
#         if x < min_:
#             min_ = x
#     return min_

# def max_(data):
#     max_ = float('-inf')
#     for x in data:
#         if x > max_:
#             max_ = x
#     return max_


data = pd.read_csv('datasets/dataset_train.csv')
print(describe(data).T)



# x = np.arange(1, 11)
# print(x)
# stat = stats(x)
# print(f'count: {stat["count"]}')
# print(f'mean: {stat["mean"]}')
# print(f'std: {stat["std"]}')
# quartiles = quartile(x)
# print(f'min: {quartiles[0]}')
# print(f'25%: {quartiles[25]}')
# print(f'50%: {quartiles[50]}')
# print(f'75%: {quartiles[75]}')
# print(f'max: {quartiles[100]}')
# print(pd.DataFrame(x).describe())