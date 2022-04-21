import os
import _utils
import numpy as np
import pandas as pd

from reverse_sampling import RRS_handler


class IMP:
    result_path = "./result/"
    datasets_path = "./datasets/"
    dataset_name = None
    rrs_path = None

    G = None  # graph stored as edge list
    V = None  # num of nodes
    E = None  # num of directed edge
    p = 0.2  # propagation probability
    mc = 1000  # monte carlo repetition

    weighted = False  # if graph is weighted
    directed = False  # if graph is directed

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

        self.set_k(k)

        if p is not None:
            self.p = p

        self._rrs_handler = RRS_handler(self)

    def set_k(self, k):
        if k == 0:
            self.k = self.V
        elif k < 1:
            self.k = int(self.V * k)
        else:
            self.k = k

    def load_RRS_data(self):
        """
        use it before running an algorithm
        :return:
        """
        self._rrs_handler.refresh_copy()

    def get_a_unused_rrset(self):
        """
        return a unused rr set
        :return:
        """
        return self._rrs_handler.get_a_random_rrs()

    def save_new_rrset(self):
        """
        use it before running an algorithm
        :return:
        """
        self._rrs_handler.update_file()

    def refresh_rrset(self):
        self._rrs_handler.refresh_copy()
