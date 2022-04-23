import decimal
import numpy as np

# IMM
IMM_para_epsilon = 0.5
IMM_para_l = 1


def covered_fraction(RRS, solution):
    count = 0
    solution_set = set(solution)

    # count the frequency of intersection between the seed set and every rr set
    for R in RRS:
        for r in R:
            if r in solution_set:
                count += 1
                break

    return count / len(RRS)


def IMM_node_selection(RRS, k, n):
    # nodes = set(range(1, n + 1))
    #
    # S_k = []
    # fraction_Sk = covered_fraction(RRS, S_k)
    #
    # for _ in range(k):
    #     marginal_benefit = -1
    #     v = 0  # #0 is an invalid node
    #
    #     # identify the vertex v that maximizes FR(S_k ^ v) - FR(S_k)
    #     for node in nodes:
    #         appended = S_k.copy()
    #         appended.append(node)
    #         tmp = covered_fraction(RRS, appended) - fraction_Sk
    #         if tmp > marginal_benefit:
    #             v = node
    #             marginal_benefit = tmp
    #     fraction_Sk = fraction_Sk + marginal_benefit
    #
    #     # insert v into S_k
    #     S_k.append(v)
    #     nodes.remove(v)

    # candidates = list(range(1, n + 1))
    # marg_gain = [covered_fraction(RRS, [c]) for c in candidates]
    #
    # Q = sorted(zip(candidates, marg_gain), key=lambda x: x[1], reverse=True)
    #
    # S_k, coverage, Q = [Q[0][0]], Q[0][1], Q[1:]
    #
    # for _ in range(1, k):
    #     found = False
    #
    #     while not found:
    #         current = Q[0][0]
    #         Q[0] = (current, covered_fraction(RRS, S_k + [current]) - coverage)
    #
    #         Q = sorted(Q, key=lambda x: x[1], reverse=True)
    #
    #         found = Q[0][0] == current
    #
    #     S_k.append(Q[0][0])
    #     coverage += Q[0][1]
    #     Q = Q[1:]

    Sk = set()
    rr_degree = [0 for ii in range(n + 1)]
    node_rr_set = dict()
    # node_rr_set_copy = dict()
    matched_count = 0
    for j in range(0, len(RRS)):
        rr = RRS[j]
        for rr_node in rr:
            # print(rr_node)
            rr_degree[rr_node] += 1
            if rr_node not in node_rr_set:
                node_rr_set[rr_node] = list()
                # node_rr_set_copy[rr_node] = list()
            node_rr_set[rr_node].append(j)
            # node_rr_set_copy[rr_node].append(j)
    for i in range(k):
        max_point = rr_degree.index(max(rr_degree))
        Sk.add(max_point)
        matched_count += len(node_rr_set[max_point])
        index_set = []
        for node_rr in node_rr_set[max_point]:
            index_set.append(node_rr)
        for jj in index_set:
            rr = RRS[jj]
            for rr_node in rr:
                rr_degree[rr_node] -= 1
                node_rr_set[rr_node].remove(jj)
    # return Sk, matched_count / len(RRS)

    return Sk


def IMM_sampling(imp):
    # initialize a set R = None and a integer LB = 1
    RRS = []
    LB = 1

    _n = imp.V
    _k = imp.k

    # let epsilon = sqrt(2) * epsilon
    _epsilon_prime = np.sqrt(2) * IMM_para_epsilon

    # base on equation 9, calculate lambda
    log_comb_n_k = float(str(decimal.Decimal(np.math.comb(_n, _k)).ln()))
    _lambda_prime = (2 + 2 / 3 * _epsilon_prime) * (log_comb_n_k + IMM_para_l * np.log(_n) + np.log(np.log2(_n))) * _n / (_epsilon_prime ** 2)

    for i in range(1, int(np.log2(_n))):
        print(i)

        x = _n / np.power(2, i)
        theta_i = _lambda_prime / x

        print("theta_i is {0}".format(theta_i))
        while len(RRS) <= theta_i:
            # select a node v from G uniformly at random
            # generate an RR set for v, and insert it into R
            RRS.append(imp.reverse_sample_IC())

        Si = IMM_node_selection(RRS, _k, _n)
        FrSi = covered_fraction(RRS, Si)
        if _n * FrSi >= (1 + _epsilon_prime) * x:
            LB = _n * FrSi / (1 + _epsilon_prime)
            break

    # lambda star is defined in Equation 6
    _alpha = np.sqrt(IMM_para_l * np.log(_n) + np.log(2))
    _beta = np.sqrt((1 - 1 / np.math.e) * (log_comb_n_k + IMM_para_l * np.log(_n) + np.log(2)))
    _lambda_star = 2 * _n * (((1 - 1 / np.math.e) * _alpha + _beta) ** 2) * (IMM_para_epsilon ** -2)
    _theta = _lambda_star / LB
    print("_theta is {0}".format(_theta))
    while len(RRS) <= _theta:
        # select a node v from G uniformly at random
        # generate an RR set for v, and insert it into R
        RRS.append(imp.reverse_sample_IC())

    return RRS


def IMM(imp):
    global IMM_para_l
    IMM_para_l0 = IMM_para_l

    IMM_para_l = IMM_para_l * (1 + np.log(2) / np.log(imp.V))
    RRS = IMM_sampling(imp)
    S_k = IMM_node_selection(RRS, imp.k, imp.V)

    IMM_para_l = IMM_para_l0

    return list(S_k)


def IMM_points(imp,  cnt):
    points = []

    step = int(imp.k / cnt)
    if step == 0:
        step = 1
    k = imp.k
    k0 = k

    while k > 0:
        imp.set_k(k)
        imp.refresh_rrset()

        S_k, RRS = IMM(imp)

        points.append(S_k)
        k = k - step

    imp.set_k(k0)
    return points


def save_solution(solution, func_name,  imp):
    """

    :param func_name:
    :param imp:
    :param solution: [ [influences], [costs]]
    :return:
    """

    # f = open()
    pass

