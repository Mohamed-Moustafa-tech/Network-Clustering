#from load_data import data_preprocessing
import networkx as nx
import pandas as pd
import numpy as np

from results_processing import results_analysis
from libAP import LSOprimizer

#true_case = open("data/true_HT.txt", "r").read().split("\n")[:-1]
#true_control = open("data/true_control.txt", "r").read().split("\n")[:-1]

path_expr, path_net = 'Sample.txt', 'IID_LUNG.graphml'

#GE, GX, labels_ids, rev = data_preprocessing(path_expr, path_net, log2=True, size=3000)


# load Gene expression 
df = pd.read_csv(path_expr, sep='\t')
df = df.set_index('Geneid')
df= df.transpose()
GE= df.to_numpy().reshape(71,2,58051)
pat = np.array(list(df.columns))
#print(pat,pat.ndim)

#n, m = GE.shape
# load network
G = nx.read_graphml(path_net)

T = 20
alpha = 0.01  # speed of temperature decrease
L_min = 30
L_max = 40

optimizer = LSOprimizer(GE, G,pat, T, L_min, L_max, max_iter=20)

nodes, labels, sc = optimizer.run_ls()