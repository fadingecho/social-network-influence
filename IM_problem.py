import os
import random

import _utils
import numpy as np
import pandas as pd


class IMP:
    result_path = "./result/"
    datasets_path = "./datasets/"
    dataset_name = None
    rrs_path = None

    G = None  # graph stored as edge list
    G_transpose = None  # for reverse sample
    V = None  # num of nodes
    E = None  # num of directed edge
    p = 0.2  # propagation probability
    mc = 1000  # monte carlo repetition

    weighted = False  # if graph is weighted

    k = 0  # k-seed users in IMP

    def __init__(self, dataset_name, k=50, p=0.2, mc=1000):
        self.mc = mc
        self.p = p

        # read graph info
        self.dataset_name = dataset_name
        self.rrs_path = self.result_path + self.dataset_name + "/RRS-out" + "{:.1}".format(self.p) + ".txt"
        G_info = pd.read_csv(
            self.datasets_path + dataset_name + '.txt',
            index_col=False,
            names=['V', 'E'],
            nrows=1,
            delimiter="\t",
        )
        self.V = G_info.iat[0, 0]
        self.E = G_info.iat[0, 1]
        self.G = pd.read_csv(
            self.datasets_path + dataset_name + '.txt',
            delimiter="\t",
            index_col=False,
            names=['source', 'target'],
            dtype='Int32',
            skiprows=1
        )
        self.G_transpose = self.G.loc[:, ['target', 'source']]
        self.G_transpose.columns = ['source', 'target']
        self.set_k(k)

        if p is not None:
            self.p = p

    def set_k(self, k):
        if k == 0:
            self.k = self.V
        elif k < 1:
            self.k = int(self.V * k)
        else:
            self.k = k

    def reverse_sample_IC(self, node=None):
        if node is None:
            node = random.choice(list(range(1, self.V + 1)))

        S = [node]
        reverse_spread = []
        # Simulate propagation process
        new_active, A = S[:], S[:]
        while new_active:
            # Get edges that flow out of each newly active node
            temp = self.G_transpose.loc[self.G_transpose['source'].isin(new_active)]

            # Extract the out-neighbors of those nodes
            targets = temp['target'].tolist()

            success = np.random.uniform(0, 1, len(targets)) < self.p

            # Determine those neighbors that become infected
            new_ones = np.extract(success, targets)

            # Create a list of nodes that weren't previously activated
            new_active = list(set(new_ones) - set(A))

            # Add newly activated nodes to the set of activated nodes
            A += new_active

        return A
