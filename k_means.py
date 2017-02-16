import csv
import numpy as np
from sklearn.cluster import KMeans

import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib

def read_data(filename):
    csv_file = open(filename,'rU')
    rows = csv.reader(csv_file)
    data = [row for row in rows]
    csv_file.close()
    return data

descriptor_data = np.genfromtxt('bow_csv.csv',delimiter=";",dtype=int)

descriptor_data = descriptor_data[:,0:128]

km = KMeans(n_clusters=100)

km.fit(descriptor_data)

joblib.dump(km,"k_means_model.pkl")
_


