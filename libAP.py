import numpy as np
import operator
#from sklearn.cluster import KMeans
import graph_tool.all as gt
from utils import *

from tslearn.clustering import TimeSeriesKMeans
import tslearn

flatten = lambda l: [item for sublist in l for item in sublist]


class LSOprimizer:
    def __init__(self, GE, G, L_min, L_max, T = 20, max_iter=100, plot=True, opt_pat=None, k=2,
                 init_size=None, seed=None, verbose=True):
        """
        Given a graph G and gene expression matrix GE finds the optimal subnetwork in G of size at least L_min and
        at most L_max that can provide the optimal patients clustering in k clusters.
        :param GE: pandas DatFrame with gene expression
        :param G: networkX graph with PPI network
        :param L_min: minimal desired solution subnetwork size
        :param L_max: maximal desired solution subnetwork size
        :param T: temprature parameter for SA
        :param max_iter: maximal allowed number of iterations
        :param plot: convergence plot (True/False)
        :param opt_pat: patients labels (if provided, patients clustering won't be performed
        :param k: nmber of clusters
        :param init_size: initial subnetwork size (default L_max *2)
        :param seed: seed
        :param verbose: True/False
        """
        self.G = nx2gt(G)
        self.T = T
        self.L_min = L_min
        self.L_max = L_max
        self.max_iter = max_iter
        self.plot = plot
        if opt_pat is None:
            self.opt_pat = []
        else:
            self.opt_pat = opt_pat
        self.k = k
        if init_size is None:
            self.init_size = L_max * 2
        else:
            self.init_size = init_size
        self.seed = seed
        self.verbose = verbose
        self.ge = GE.values
        self.genes = list(self.G.get_vertices())
        self.patients = np.array(list(GE.columns))

    def APUtil(self, u, visited, ap, parent, low, disc, nodes, Time=0):
        """
        A recursive function that find articulation points
        using DFS traversal
        :param u: the vertex to be visited next
        :param visited: keeps tract of visited vertices
        :param ap: stores articulation points
        :param parent: stores parent vertices in DFS tree
        :param low: low value
        :param disc: stores discovery times of visited vertices
        :param nodes: current node set

        for more details: https://www.geeksforgeeks.org/articulation-points-or-cut-vertices-in-a-graph/
        """

        # Count of children in current node
        children = 0

        # Mark the current node as visited and print it
        visited[u] = True

        # Initialize discovery time and low value
        disc[u] = Time
        low[u] = Time
        Time += 1

        # for all the vertices adjacent to this vertex
        for v in self.G.vertex(u).out_neighbors():
            # If v is not visited yet, then make it a child of u
            # in DFS tree and recur for it
            if int(v) in nodes:
                if not visited[int(v)]:
                    parent[int(v)] = int(u)
                    children += 1
                    self.APUtil(int(v), visited, ap, parent, low, disc, nodes, Time)

                    # Check if the subtree rooted with v has a connection to
                    # one of the ancestors of u
                    low[u] = min(low[u], low[int(v)])

                    # u is an articulation point in following cases
                    # (1) u is root of DFS tree and has two or more children.
                    if parent[u] == -1 and children > 1:
                        ap[u] = True

                    # (2) If u is not root and low value of one of its child is more
                    # than discovery value of u.
                    if parent[u] != -1 and low[int(v)] >= disc[u]:
                        ap[u] = True

                        # Update low value of u for parent function calls
                elif int(v) != parent[u]:
                    low[u] = min(low[u], disc[int(v)])

                    # The function to do DFS traversal. It uses recursive APUtil()

    def is_AP(self, nodes):
        """
        Checks which nodes in the given set of nodes can NOT be removed without breaking
        disconnecting the induced subgraph
        :param nodes: set of nodes that make an induced subgraph of G
        :return: dictionary where each key is a node and each value indicates if a node is
        removable (articulation point)
        """
        visited = dict()
        disc = dict()
        low = dict()
        parent = dict()
        ap = dict()
        for node in nodes:
            visited[node] = False
            disc[node] = float("Inf")
            low[node] = float("Inf")
            parent[node] = -1
            ap[node] = False

        # Call the recursive helper function
        # to find articulation points
        # in DFS tree rooted with vertex 'i'
        for node in nodes:
            if not visited[node]:
                self.APUtil(node, visited, ap, parent, low, disc, set(nodes))

        return ap

    def score(self, nodes, labels):
        """
        scores  given solution which is defined as a subnetwork and patient clusters
        :param nodes: list of nodes used in the solution
        :param labels: patient cluster labels
        :return: objective function value
        """
        #Compute distances to center candidates
        Distance = tslearn.cdist_dtw(self.ge[labels],self.ge)
        inertia = Distance.min(axis=1).sum()
        return inertia
   
        
        #vs = []
        #centroids = []
        #for i in range(self.k):
         #   idx = np.asarray(labels == i).nonzero()[0]
          #  vals = np.mean(self.ge[np.ix_(nodes, idx)], axis=1)
           # centroids.append(np.mean(self.ge[np.ix_(nodes, idx)]))
           # vs.append(vals)
        #objective = []
        #for i in range(self.k):
        #    dif = np.mean(np.power((vs[i] - centroids[i]), 2))
         #   objective.append(dif)

        #return np.mean(objective)



    def dfs(self, node, d, visited=None):
        """
        Recursive DFS
        :param node: starting node
        :param d: length of s required subnetwork
        :param visited: should be left empty
        :return: a list of connected nodes of length d
        """

        if visited is None:
            visited = []
        if int(node) not in visited and len(visited) < d:
            visited.append(int(node))
            for neighbour in node.out_neighbors():
                self.dfs(neighbour, d, visited)
        if len(visited) == d:
            return visited

    def get_candidates(self, nodes):
        """
        Outputs first-degree neighbours of given nodes in graph G
        :param nodes: list of nodes that form a subnetwork/solution
        :return: list of first neighbour nodes
        """
        subst_candidates = flatten([[int(n) for n in self.G.get_all_neighbours(x)] for x in nodes])
        subst_candidates = set(subst_candidates).difference(set(nodes))
        return subst_candidates

    def insertion(self, nodes, labels):
        """
        Scores all possible insertions
        :param nodes: current solution
        :param labels: patient clusters labels
        :return: dictionary where key are possible insertions and values are scores
        """
        results = dict()
        size = len(nodes)
        if size < self.L_max:
            candidates = self.get_candidates(nodes)
            for c in candidates:
                nodes_new = nodes + [c]
                sc = self.score(nodes_new, labels)
                results["i_" + str(c)] = sc
        return results

    def deletion(self, nodes, labels, AP):
        """
        Scores all possible deletions
        :param nodes: current solution
        :param labels: patient clusters labels
        :param AP: articulation points (can't be removed since they separate the subnetwork)
        :return: dictionary where key are possible deletions and values are scores
        """
        results = dict()
        size = len(nodes)

        if size > self.L_min:
            for node in nodes:
                if not AP[node]:
                    nodes_new = list(set(nodes).difference({node}))
                    sc = self.score(nodes_new, labels)
                    results["d_" + str(node)] = sc
        return results

    def subst(self, nodes, labels, AP):
        """
        Scores all possible substitutions
        :param nodes: current solution
        :param labels: patient clusters labels
        :param AP: articulation points (can't be removed since they separate the subnetwork)
        :return: dictionary where key are possible substitutions and values are scores
        """
        results = dict()
        size = len(nodes)
        if (size < self.L_max) and (size > self.L_min):
            for node in nodes:
                without_node = set(nodes) - {node}
                candidates = self.get_candidates(list(without_node))
                candidates = candidates - {node}
                for c in candidates:
                    if AP[node]:
                        nodes_new = list(without_node.union({c}))
                        if self.is_connected(nodes_new):
                            sc = self.score(nodes_new, labels)
                            results["s_" + str(node) + "_" + str(c)] = sc

                    else:
                        nodes_new = list(without_node.union({c}))
                        sc = self.score(nodes_new, labels)
                        results["s_" + str(node) + "_" + str(c)] = sc

        return results

    def is_connected(self, nodes):
        """
        Checks if a subgraph of G that consists of the given nodes is connected
        :param nodes: list of nodes
        :return: bool
        """
        sg = self.G.new_vertex_property("bool")
        for node in nodes:
            sg[node] = True
        g = gt.GraphView(self.G, vfilt=sg)

        comp, _ = gt.label_components(g, vprop=sg)
        if len(set(comp.a[nodes])) > 1:
            return False
        else:
            return True


    @staticmethod
    def do_action_nodes(action, nodes):
        """
        Updates the set of nodes given the action
        :param action: a key from the results dictionary that has a description of an action
        :param nodes: previous solution
        :return: new set of nodes
        """
        if len(action.split("_")) == 2:  # inserion or deletion
            act, node = action.split("_")
            node = int(node)
            if act == "i":
                nodes = nodes + [node]
            else:
                nodes = list(set(nodes).difference({node}))
        else:  # substitution
            act, node, cand = action.split("_")
            node = int(node)
            cand = int(cand)
            nodes = nodes + [cand]
            nodes = list(set(nodes).difference({node}))
        return nodes

    @staticmethod
    def to_key(nodes):
        """
        Creates a string representation of nodes
        :param nodes: node list
        :return: string of nodes
        """
        nodes = sorted(nodes)
        nodes = [str(node) for node in nodes]
        nodes = "|".join(nodes)
        return nodes

    @staticmethod
    def do_action_patients(action, labels):
        """
        Modifies patient cluster labels according to the given action
        :param action: a key from results dictionary
        :param labels: old cluster labels
        :return: updated cluster labels
        """
        if len(action.split("_")) == 2:  # add patient to a group
            _, group, idx = action.split("_")
            idx = int(idx)
            group = int(group)
            labels[idx] = group
        else:  # substitution
            _, idx1, idx2 = action.split("_")
            idx1 = int(idx1)
            idx2 = int(idx2)
            old = labels[idx1]
            labels[idx1] = labels[idx2]
            labels[idx2] = old
        return labels

    def ls_on_genes(self, nodes, labels, solutions, score0, T):
        """
        Runs local search on a gene set
        :param nodes: current node set
        :param labels: current patient clusters lables
        :param solutions: dictionary wth previously used solutions
        :param score0: last objective function score
        :param T: temperature for SA

        :return:
        nodes - new set of nodes
        score1 - new score
        move - True if further optimization was possible
        """
        # SUBNETWORK OPTIMIZATION
        move = False  # indicates if the solution feasible
        AP = self.is_AP(nodes)
        results = {**self.insertion(nodes, labels), **self.deletion(nodes, labels, AP),
                   **self.subst(nodes, labels, AP)}
        # first select the highest scoring solution which doesn't lead to the same set of nodes
        while not move:
            action = max(results.items(), key=operator.itemgetter(1))[0]
            score1 = results[action]
            # check if the solution is feasible
            nodes_new = self.do_action_nodes(action, nodes)
            nodes_new = self.to_key(nodes_new)
            if solutions.get(nodes_new) == None:  # solution wasn't used before
                move = True
            else:
                del results[action]
                if len(results) == 0:
                    print("no more feasible solutions")
                    return nodes, score0, move

        delta = score0 - score1
        if delta < 0:  # move on
            print(action)
            print("Score after genes LS {0}".format(score1))
            nodes = self.do_action_nodes(action, nodes)

        else:  # SA
            try:
                val = np.exp(-delta / T)
            except RuntimeError:
                val = 0
            p = np.random.uniform()
            if val > p:  # move on
                print("SA on genes at {0} degrees".format(T))
                print(action)
                print("Score after genes LS".format(score1))
                nodes = self.do_action_nodes(action, nodes)
            else:  # terminate if no improvement in two rounds
                print("too cold for genes SA, no actions taken")
                move = False
                score1 = score0
        return nodes, score1, move

    def ls_on_patients(self, nodes):
        """

        :param nodes: current node set

        :return:
        labels - new patient clusters labels
        score1 - updated objective function score
        """
        seed = 0
        labels = TimeSeriesKMeans(n_clusters=2,
                          n_init=2,
                          metric="dtw",
                          max_iter_barycenter=10,
                          random_state=seed).fit_predict(self.ge[nodes, :].T)

        # PARTITION OPTIMIZATION
        # kmeans = KMeans(n_clusters=self.k, random_state=0).fit(self.ge[nodes, :].T)
        #     else:
        #         centroids = []
        #         for i in range(k):
        #             idx = np.asarray(labels0 == i).nonzero()[0]
        #             vals = np.mean(ge[np.ix_(nodes, idx)], axis = 1)
        #             centroids.append(vals)
        # #            print(vals)
        #         kmeans = KMeans(n_clusters=k, random_state=0, init = np.array(centroids)).fit(ge[nodes, :].T)
        # labels = kmeans.labels_
        score1 = self.score(nodes, labels)
        print("Reclustered score {0}".format(score1))
        return labels, score1

    def run_ls(self):
        """
        Runs LS on patients and nodes
        :return:
        best_nodes - optimized node set
        best_labels - optimized label set
        score_max -maximal score
        """

        T0 = self.T
        T = T0
        score_max = 0
        best_nodes = []
        best_labels = []
        n, m = self.ge.shape
        pats = self.patients - n
        # initialization
        if self.seed is None:
            nodes = []
            # for  whatever reason dfs sometimes returns nothing
            no_type = True
            while no_type:
                nodes = self.dfs(self.G.vertex(np.random.choice(self.genes, 1)[0]), self.init_size)
                if nodes is not None:
                    no_type = False
        else:
            nodes = self.seed
        if len(self.opt_pat) != m:
            labels = np.random.choice([0, 1], len(pats))
        else:
            labels = np.array(self.opt_pat)
        start_score = self.score(nodes, labels)
        if self.verbose:
            print(start_score)
        score0 = start_score
        scores = [start_score]
        solutions = dict()
        nodes_keys = self.to_key(nodes)
        solutions[nodes_keys] = ""
        count = 0
        for it in range(self.max_iter):
            if len(self.opt_pat) != m:

                if count != 0:
                    labels, score1 = self.ls_on_patients(nodes)
                else:
                    labels, score1 = self.ls_on_patients(nodes)

            else:
                score1 = score0
            nodes_backup = nodes
            labels_backup = labels
            nodes, score2, move_genes = self.ls_on_genes(nodes, labels, solutions, score1, T)
            if not self.is_connected(nodes):
                print("something is wrong, network is disconnected")
                return nodes_backup, labels_backup, 0
            T = T0 * (0.9 ** it)  # cool down
            if self.verbose:
                print(it)
            solutions[self.to_key(nodes)] = ""
            scores.append(score2)
            if self.plot:
                convergence_plot(scores)

            score0 = score2
            if score2 > score_max:
                score_max = score2
                best_nodes = nodes
                best_labels = labels
            count = count + 1
            if not move_genes:
                break
        return best_nodes, best_labels, score_max

