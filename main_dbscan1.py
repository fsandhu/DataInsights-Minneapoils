import numpy as np
import pandas as pd
from sklearn import metrics

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt# reading the classic iris dataset into a df
st_df = pd.read_csv("C://Users//Nithish//Downloads//CSCE474Project-main//CSCE474Project-main//forceDataPreProcessed.csv")
dataset = st_df.head(n=1000)
print(dataset)
dataset_table = dataset[['PoliceUseOfForceID','CaseNumber','X', 'Y']]
print(dataset_table)
dataset_table.columns  = ['PoliceUseOfForceID','CaseNumber','X', 'Y']
sp_th = 600
tem_th = 60
min_neighbors = 40 # points
st_dbscan_m = DBSCAN((dataset_table, sp_th, tem_th, min_neighbors))
print(st_dbscan_m)
labels = dataset['Is911Call'].values
X_main = dataset_table[['X', 'Y']].values
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
            # Black Used for noise. The cluster results are -1 is noise point, expressed as black
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X_main[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=6)

    plt.title('DBSCAN: #n of clusters {}'.format(len(unique_labels)))
    plt.xlabel('longitude(E)')
    plt.ylabel('latitude(N)')
    plt.show()
