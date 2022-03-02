import os

import numpy as np
import pandas as pd

para_l = 1
para_epsilon = 0.3


class my_IM:
    result_path = "./result/"
    datasets_path = "./datasets/"
    dataset_name = None
    G = None
    V = None
    E = None

    p = 0.5
    RRS = None

    weighted = False
    directed = False

    k = None
    theta = None

    def __init__(self, dataset_name, k=50, weighted=False, directed=False):
        self.dataset_name = dataset_name
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

        self.k = k
        self.weighted = weighted
        self.directed = directed

        if directed:
            print("this data is undirected")
            G_copy = self.G.copy(deep=True)
            G_copy[['target', 'source']] = G_copy[['source', 'target']]
            G_copy.columns = ['source', 'target']
            self.G = pd.concat([self.G, G_copy], ignore_index=True)

            self.E = self.E * 2

        self.theta = self.get_theta(p=self.p)
        self.RRS = self.read_RRS()
        if abs(len(self.RRS) - self.theta) > 3000:
            self.RRS = self.generate_RRS(p=self.p)
            self.save_RRS()

    def read_RRS(self):
        RRS = []
        RRS_file = open(self.result_path + self.dataset_name + "/RRS-out.txt")
        for rf in RRS_file.readlines():
            rf = rf.strip('[]\n')
            RRS.append(list(map(int, rf.split(","))))
        print("read RRS from file, size : " + str(len(RRS)))

        return RRS

    def save_RRS(self):
        try:
            np.savetxt(self.result_path + self.dataset_name + "/RRS-out.txt", self.RRS, delimiter=", ", fmt="% s")
        except FileNotFoundError:
            os.makedirs(self.result_path + self.dataset_name)
            np.savetxt(self.result_path + self.dataset_name + "/RRS-out.txt", self.RRS, delimiter=", ", fmt="% s")

    def get_random_RRS(self, p=0.5):
        """
        Inputs: G:  Ex2 dataframe of directed edges. Columns: ['source','target']
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

    def generate_RRS(self, p=0.5):
        RRS = []

        print("creating RRS")
        print("\r{:3}%".format(0), end="")
        for _i in range(0, self.theta):
            r = self.get_random_RRS(p=p)
            RRS.append(r)
            print("\r{:3}%".format((_i + 1) / self.theta * 100), end="")
        print("\n")

        return RRS

    def IC(self, S, p=0.5, mc=10000):
        """
        Input:  G:  Ex2 dataframe of directed edges. Columns: ['source','target']
                S:  Set of seed nodes
                p:  Disease propagation probability
                mc: Number of Monte-Carlo simulations
        Output: Average number of nodes influenced by the seed nodes
        """

        # Loop over the Monte-Carlo Simulations
        spread = []
        for _ in range(mc):

            # Simulate propagation process
            new_active, A = S[:], S[:]
            while new_active:
                # Get edges that flow out of each newly active node
                temp = self.G.loc[self.G['source'].isin(new_active)]

                # Extract the out-neighbors of those nodes
                targets = temp['target'].tolist()

                success = np.random.uniform(0, 1, len(targets)) < p

                # Determine those neighbors that become infected
                new_ones = np.extract(success, targets)

                # Create a list of nodes that weren't previously activated
                new_active = list(set(new_ones) - set(A))

                # Add newly activated nodes to the set of activated nodes
                A += new_active

            spread.append(len(A))

        return np.mean(spread)

    def width_of_RR_set(self, R):
        # TODO need to test it
        # TODO maybe a table of income degree
        vc = self.G['target'].value_counts()
        width = 0
        for node in R:
            try:
                width = width + vc[node]
            except KeyError:
                width = width

        return width

    def KPT_estimation(self, p=0.5):
        G = self.G
        n = self.V
        m = self.E

        for i in range(1, int(np.log2(n) - 1)):
            ci = 6 * para_l * np.log(n) + 6 * np.log(np.log2(n)) * np.exp2(i)
            _sum = 0
            for j in range(1, int(ci)):
                R = self.get_random_RRS(p=p)
                kR = 1 - (1 - self.width_of_RR_set(R) / m) ** self.k
                _sum = _sum + kR
            if _sum / ci > 1 / np.exp2(i):
                return n * _sum / (2 * ci)
        return 1

    def get_theta(self, p=0.5):
        kpt = self.KPT_estimation(p=p)
        _lambda = (8 + 2 * para_epsilon) * self.V * (
                para_l * np.log(self.V) + np.log(float(np.math.comb(self.V, self.k))) + np.log(2)) * para_epsilon ** (
                      -2)
        theta = int(_lambda / kpt)
        return theta
