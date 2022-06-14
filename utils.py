import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import graph_tool.all as gt
import pandas as pd

def nx2gt(G):
    """
    Converts a networkx graph to a graph-tool graph.
    """
    d = nx.to_dict_of_lists(G)
    edges = [(i, j) for i in d for j in d[i]]
    edges_sorted = []
    for e in edges:
        if e[1] < e[0]:
            edges_sorted.append((e[1], e[0]))
        else:
            edges_sorted.append(e)
    edges_sorted = list(set(edges_sorted))

    GT = gt.Graph(directed=False)# from lifelines import KaplanMeierFitter
# from lifelines.statistics import logrank_test
    GT.add_edge_list(edges_sorted)
    return GT


def jac(x, y):
    """
    Jaccard index between list x and list y
    """
    if len(x) > 0 and len(y) > 0:
        return len(set(x).intersection(set(y))) / len((set(x).union(set(y))))
    else:
        return 0


def convergence_plot(scores):
    """
    Shows the convergence plot

    Attributes:
    -----------
    scores - the output of run_search() function

    output - directory name where results should be saved
    """

    plt.figure(figsize=(10, 6))

    sns.set(style="whitegrid")
    plt.rc('font', size=13)  # controls default text sizes
    plt.rc('axes', titlesize=13)  # fontsize of the axes title
    plt.rc('xtick', labelsize=13)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=13)  # fontsize of the tick labels
    plt.rc('legend', fontsize=13)
    wg = pd.DataFrame(scores, columns=["score"])
    ax = sns.lineplot(data=wg, palette="tab10", linewidth=2.5)
    ax.set(xlabel="Iterations")
    ax.set(ylabel="Score")
    plt.show()

    # def plot_survival(col, clinical, labels=None, ci_show=True, show_img=True):
    #     values = list(set(clinical[col]))
    #     if labels == None:
    #         labels = values
    #         labels = [str(x) for x in values]
    #     kmfs = []
    #     T = []
    #     labels_true = []
    #     C = []
    #     count = 0
    #     for val in values:
    #         v = str(val)
    #         km = KaplanMeierFitter()
    #         f = clinical[col] == val
    #         t = clinical[f]['surv']
    #         if t.shape[0] > 0:
    #             print("{0} group {1} sampels".format(labels[count], t.shape[0]))
    #
    #             c = clinical[f]['event']
    #             kmfs.append(km)
    #             T.append(t)
    #             C.append(c)
    #             labels_true.append(labels[count])
    #         count = count + 1
    #     plt.figure(figsize=(12, 8))
    #     ax = plt.subplot(111)
    #     for i in range(len(C)):
    #         kmf = kmfs[i]
    #         t = T[i]
    #         c = C[i]
    #         label = labels[i]
    #         kmf.fit(t, event_observed=c, label=labels_true[i])
    #         if show_img:
    #             kmf.plot(ci_show=ci_show, ax=ax)
    #     if show_img:
    #         plt.show()
    #     p_vals = []
    #     for i in range(len(C)):
    #         for j in range(i + 1, len(C)):
    #             summary = logrank_test(T[i], T[j], C[i], C[j], alpha=99)
    #             v = round(-np.log10(summary.p_value), 2)
    #             print("- log10 p-value between {0} and {1} is {2}".format(labels[i], labels[j], v))
    #             p_vals.append(v)
    #     print("average -log10 p-value is {0}".format(np.round(np.mean(p_vals), 2)))
    #     return (np.round(np.mean(p_vals), 2))

