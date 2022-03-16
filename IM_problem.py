import os
import _utils
import numpy as np
import pandas as pd

para_l = 1
para_epsilon = 0.5


def width_of_RR_set(R, in_degrees):
    """
        Inputs: R : a random rr set
                in_degrees : in_degrees of nodes sorted by degree
        Return: width of R, sum of in-degrees
    """
    width = 0
    for node in R:
        try:
            width = width + in_degrees[node]
        except KeyError:
            width = width

    return width


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
    theta = None  # theta in TIM

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

    def load_RRS(self):
        # if the existing rrs file is not large enough, extend to a new one
        # else we just shrink the file to fit theta
        self.RRS = self.read_RRS()
        # get_theta() needs RRS to estimate KPT, if RRS is exhausted, new rrset will be appended
        self.theta = self.get_theta(p=self.p)
        print("theta is " + str(self.theta))

        if len(self.RRS) - self.theta < -100:
            self.RRS.extend(self.generate_RRS(num=self.theta - len(self.RRS), p=self.p))
            self.save_RRS()
        elif len(self.RRS) - self.theta > 100:
            self.RRS = self.RRS[:self.theta]

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

    def get_a_random_RRS(self, p=0.5):
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

    def generate_RRS(self, num=None, p=0.5):
        """
            Return: a list of RRS, len = num
        """
        if num is None:
            num = self.theta
        RRS = []

        _utils.show_process_bar("reverse sampling :", 0, num)
        for _i in range(0, num):
            r = self.get_a_random_RRS(p=p)
            RRS.append(r)
            _utils.show_process_bar("reverse sampling :", _i, num)
        _utils.process_end("")
        return RRS

    def IC(self, S):
        """
        Input:  S:  Set of seed nodes
        Output: Average number of nodes influenced by the seed nodes
        """
        # Loop over the Monte-Carlo Simulations
        spread = []
        for _ in range(self.mc):

            # Simulate propagation process
            new_active, A = S[:], S[:]
            while new_active:
                # Get edges that flow out of each newly active node
                temp = self.G.loc[self.G['source'].isin(new_active)]

                # Extract the out-neighbors of those nodes
                targets = temp['target'].tolist()

                success = np.random.uniform(0, 1, len(targets)) < self.p

                # Determine those neighbors that become infected
                new_ones = np.extract(success, targets)

                # Create a list of nodes that weren't previously activated
                new_active = list(set(new_ones) - set(A))

                # Add newly activated nodes to the set of activated nodes
                A += new_active

            spread.append(len(A))

        return np.mean(spread)

    def KPT_estimation(self, p=0.5):
        """
        refer to TIM
        :param p: propagation probability
        :return: KPT star
        """
        KPT_star = 1
        G = self.G
        n = self.V
        m = self.E

        in_degrees = G['target'].value_counts()
        RRS_it = iter(self.RRS)
        it_exhausted = False

        for i in range(1, int(np.log2(n) - 1)):
            ci = 6 * para_l * np.log(n) + 6 * np.log(np.log2(n)) * np.exp2(i)
            _sum = 0
            for j in range(1, int(ci)):
                # make use of the rrset in the file
                if it_exhausted:
                    R = self.get_a_random_RRS(p=p)
                    self.RRS.append(R)
                else:
                    try:
                        R = next(RRS_it)
                    except StopIteration:
                        it_exhausted = True
                        R = self.get_a_random_RRS(p=p)
                        self.RRS.append(R)

                kR = 1 - (1 - width_of_RR_set(R, in_degrees) / m) ** self.k
                _sum = _sum + kR

            if _sum / ci > 1 / np.exp2(i):
                KPT_star = n * _sum / (2 * ci)
                break

        self.save_RRS()
        return KPT_star

    def get_theta(self, p=0.5):
        """
        refer to TIM
        :param p: propagation probability
        :return: theta in TIM
        """
        kpt = self.KPT_estimation(p=p)

        # because of combination number explosion, treat the comb item specially
        comb_Vk = np.math.comb(self.V, self.k)
        try:
            log_comb_Vk = np.log(float(comb_Vk))
        except OverflowError:
            print("overflow")
            log_comb_Vk = 1.0
            big_float = 1e100
            tmp = comb_Vk
            log_big_float = np.log(big_float)
            while tmp > int(big_float):
                log_comb_Vk = log_comb_Vk * log_big_float
                tmp = tmp - int(big_float)
                print(str(log_comb_Vk))
            log_comb_Vk = log_comb_Vk * np.log(tmp)

        _lambda = (8 + 2 * para_epsilon) * self.V * \
                  (para_l * np.log(self.V) + log_comb_Vk + np.log(2)) * \
                  para_epsilon ** (-2)
        theta = int(_lambda / kpt)
        return theta
