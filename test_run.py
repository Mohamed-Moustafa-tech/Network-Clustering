from load_data import data_preprocessing
from results_processing import results_analysis
from libAP import LSOprimizer

true_case = open("data/true_HT.txt", "r").read().split("\n")[:-1]
true_control = open("data/true_control.txt", "r").read().split("\n")[:-1]

path_expr, path_net = 'data/expr.csv', 'data/biogrid_net.tsv'

GE, GX, labels_ids, rev = data_preprocessing(path_expr, path_net, log2=True, size=3000)
n, m = GE.shape

patients = list(GE.columns)
true_pat = [1 if labels_ids[x] in true_case else 0 for x in patients]
T = 20
alpha = 0.01  # speed of temperature decrease
L_min = 30
L_max = 40

optimizer = LSOprimizer(GE, GX, T, L_min, L_max, max_iter=20)

nodes, labels, sc = optimizer.run_ls()

results = results_analysis(nodes, labels_ids, labels, n, convert=True, origID="entrezgene")
results.show_clustermap(GE, GX, output="one_net_clustermap", true_labels=[true_case, true_control])
