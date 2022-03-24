import os
import _utils
import numpy as np
import pandas as pd


class IMP:
    result_path = "./result/"
    datasets_path = "./datasets/"
    dataset_name = None
    rrs_path = None

    G = None   # graph stored as edge list
    V = None   # num of nodes
    E = None   # num of directed edge
    p = 0.2    # propagation probability
    mc = 1000  # monte carlo repetition
    RRS = []   # random rr sets, size estimated by TIM

    weighted = False  # if graph is weighted
    directed = False  # if graph is directed

    k = 0  # k-seed users in IMP

    def __init__(self, dataset_name, k, weighted, directed, p=0.2, mc=1000):
        # read graph info
        self.dataset_name = dataset_name
        self.rrs_path = self.result_path + self.dataset_name + "/RRS-out" + "{:.1}".format(self.p) + ".txt"
        G_info = pd.read_csv(
            self.datasets_path + dataset_name + '.mtx',
            index_col=False,
            names=['in', 'out', 'num_edge'],
            skiprows=1,
            nrows=1,
            delimiter=" ",
        )
        self.V = G_info.iat[0, 0]
        self.E = G_info.iat[0, 2]
        self.G = pd.read_csv(
            self.datasets_path + dataset_name + '.mtx',
            delimiter=" ",
            index_col=False,
            names=['source', 'target'],
            dtype='Int32',
            skiprows=2
        )
        self.mc = mc
        self.p = p
        self.weighted = weighted
        self.directed = directed

        # convert undirected graph to directed
        if not directed:
            print("this network is undirected")
            G_copy = self.G.copy(deep=True)
            G_copy[['target', 'source']] = G_copy[['source', 'target']]
            G_copy.columns = ['source', 'target']
            self.G = pd.concat([self.G, G_copy], ignore_index=True)
            self.E = self.E * 2

        if k == 0:
            self.k = self.V
        elif k < 1:
            self.k = int(self.V * k)
        else:
            self.k = k

        if p is not None:
            self.p = p

    def load_RRS(self, theta):
        # if the existing rrs file is not large enough, extend to a new one
        # else we just shrink the file to fit theta
        self.RRS = self.read_RRS()
        # get_theta() needs RRS to estimate KPT, if RRS is exhausted, new rrset will be appended
        print("theta is " + str(theta))

        if len(self.RRS) - theta < -100:
            self.RRS.extend(self.generate_RRS(theta - len(self.RRS), p=self.p))
            self.save_RRS()
        elif len(self.RRS) - theta > 100:
            self.RRS = self.RRS[:theta]

    def read_RRS(self):
        RRS = []
        try:
            RRS_file = open(self.rrs_path)
        except FileNotFoundError:
            return RRS  # len(RRS) = 0

        for rf in RRS_file.readlines():
            rf = rf.strip('[]\n')
            RRS.append(list(map(int, rf.split(","))))
        print("read RRS from file, size : " + str(len(RRS)))

        return RRS

    def save_RRS(self):
        try:
            np.savetxt(self.rrs_path, self.RRS, delimiter=", ", fmt="% s")
        except FileNotFoundError:
            os.makedirs(self.result_path + self.dataset_name)
            np.savetxt(self.rrs_path, self.RRS, delimiter=", ", fmt="% s")

    def get_a_random_RRS(self, p):
        """
        Return: A random reverse reachable set expressed as a list of nodes
        """

        # Step 1. Select random source node
        source = np.random.choice(np.unique(self.G['source']))

        # Step 2. Get an instance of g
        g = self.G.copy().loc[np.random.uniform(0, 1, self.G.shape[0]) < p]

        # Step 3. Construct reverse reachable set of the random source node
        RRS = []
        new_nodes, RRS0 = [source], [source]
        while new_nodes:
            # Limit to edges that flow into the source node
            temp = g.loc[g['target'].isin(new_nodes)]

            # Extract the nodes flowing into the source node
            temp = temp['source'].tolist()

            # Add new set of in-neighbors to the RRS
            RRS = list(set(RRS0 + temp))

            # Find what new nodes were added
            new_nodes = list(set(RRS) - set(RRS0))

            # Reset loop variables
            RRS0 = RRS[:]

        return RRS

    def generate_RRS(self, num, p=0.5):
        """
            Return: a list of RRS, len = num
        """
        RRS = []

        _utils.show_process_bar("reverse sampling :", 0, num)
        for _i in range(0, num):
            r = self.get_a_random_RRS(p=p)
            RRS.append(r)
            _utils.show_process_bar("reverse sampling :", _i, num)
        _utils.process_end("")
        return RRS

