import copy
import decimal
from collections import Counter
from time import time

import bitmap
import numpy as np
import _utils
import random

from spreading_models import IC


# random


def generate_random_solution(count, imp):
    return [random_solution(imp) for _ in range(count)]


def random_solution(imp):
    return random.sample(range(1, imp.V + 1), random.randint(1, imp.V))


# CELF


def CELF(imp):
    """
    Inputs: G:  Ex2 dataframe of directed edges. Columns: ['source','target']
            p:  Disease propagation probability
    Return: greedy_trace: list[cost] = influence
    """

    # --------------------
    # Find the first node with greedy algorithm
    # --------------------
    start_time = time()

    # Compute marginal gain for each node
    candidates = np.unique(imp.G['source'])
    marg_gain = [IC(imp, [c]) for c in candidates]

    # Create the sorted list of nodes and their marginal gain
    Q = sorted(zip(candidates, marg_gain), key=lambda x: x[1], reverse=True)

    # Select the first node and remove from candidate list
    S, spread, Q = [Q[0][0]], Q[0][1], Q[1:]
    greedy_trace = [spread]

    # --------------------
    # Find the next k-1 nodes using the CELF list-sorting procedure
    # --------------------

    k = min(imp.k , len(candidates))
    _utils.show_process_bar("CELF", 1, k)
    for i in range(1, k):
        _utils.show_process_bar("CELF", i + 1, k)
        found = False

        while not found:
            # Recalculate spread of top node
            current = Q[0][0]

            # Evaluate the spread function and store the marginal gain in the list
            Q[0] = (current, IC(imp, S + [current]) - spread)

            # Re-sort the list
            Q = sorted(Q, key=lambda x: x[1], reverse=True)

            # Check if previous top node stayed on top after the sort
            found = Q[0][0] == current

        # Select the next node
        S.append(Q[0][0])
        spread = Q[0][1] + spread
        greedy_trace.append(spread)

        # Remove the selected node from the list
        Q = Q[1:]
    _utils.process_end(str(time() - start_time) + 's')

    return [greedy_trace, [_i + 1 for _i in range(len(greedy_trace))]]


# TIM
TIM_para_l = 1
TIM_para_epsilon = 0.1


def TIM_node_selection(imp, theta):
    """
        Inputs: IM_dataset:  provide estimated RRS, check IM_problem.py
        Return: greedy_trace: list[cost] = influence
    """

    # Generate theta random RR sets and insert them into set_R
    set_R = [imp.get_a_unused_rrset() for _ in range(theta)]
    set_R_bk = set_R.copy()

    # S_k is the final solution
    S_k = []
    trace = []

    _utils.show_process_bar("theta={0} TIM : ".format(theta), 0, imp.k)
    for i in range(1, imp.k + 1):
        _utils.show_process_bar("theta={0} TIM : ".format(theta), i, imp.k)

        # identify the node that covers the most RR sets in set_R
        flat_list = [item for sublist in set_R for item in sublist]
        most_common = Counter(flat_list).most_common()

        # if the recruitment cost is too large for the network
        if len(most_common) == 0:
            break

        seed = most_common[0][0]

        # add it into S_k
        S_k.append(seed)

        # remove from set_R all RR sets that are covered by it
        set_R = [rrs for rrs in set_R if seed not in rrs]

        # record trace
        trace.append(copy.deepcopy(S_k))

    _utils.process_end(" ")
    return trace, set_R_bk


def TIM_KPT_estimation(imp):
    """
    refer to TIM
    :return: KPT star
    """
    KPT_star = 1
    G = imp.G
    n = imp.V
    m = imp.E
    p = imp.p

    in_degrees = G['target'].value_counts()

    for i in range(1, int(np.log2(n) - 1)):
        ci = 6 * TIM_para_l * np.log(n) + 6 * np.log(np.log2(n)) * np.exp2(i)
        _sum = 0
        for j in range(1, int(ci)):
            R = imp.get_a_unused_rrset()

            # width of a rr set
            width = 0
            for node in R:
                try:
                    width = width + in_degrees[node]
                except KeyError:
                    width = width

            kR = 1 - (1 - width / m) ** imp.k
            _sum = _sum + kR

        if _sum / ci > 1 / np.exp2(i):
            KPT_star = n * _sum / (2 * ci)
            break

    return KPT_star


def TIM_get_theta(imp):
    """
    refer to TIM
    :return: theta in TIM
    """
    kpt = TIM_KPT_estimation(imp)

    # because of combination number explosion, treat the comb item specially
    comb_Vk = np.math.comb(imp.V, imp.k)
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

    _lambda = (8 + 2 * TIM_para_epsilon) * imp.V * (TIM_para_l * np.log(imp.V) + log_comb_Vk + np.log(2)) * TIM_para_epsilon ** (-2)
    theta = int(_lambda / kpt)
    return theta


def TIM(imp):
    """
        Inputs: IM_dataset:  provide estimated RRS, check IM_problem.py
        Return: greedy trace, a set of random rr set
    """

    theta = TIM_get_theta(imp)
    trace, RRS = TIM_node_selection(imp, theta)

    return trace, RRS


def TIM_points(imp,  cnt):
    largest_RRS = []
    points = []

    step = int(imp.k / cnt)
    if step == 0:
        step = 1
    k = imp.k
    k0 = k

    while k > 0:
        imp.set_k(k)
        trace, RRS = TIM(imp)
        if len(RRS) > len(largest_RRS):
            largest_RRS = RRS

        points.append(trace[len(trace) - 1])
        k = k - step

    imp.set_k(k0)
    return points, largest_RRS

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

    candidates = list(range(1, n + 1))
    marg_gain = [covered_fraction(RRS, [c]) for c in candidates]

    Q = sorted(zip(candidates, marg_gain), key=lambda x: x[1], reverse=True)

    S_k, coverage, Q = [Q[0][0]], Q[0][1], Q[1:]

    for _ in range(1, k):
        found = False

        while not found:
            current = Q[0][0]
            Q[0] = (current, covered_fraction(RRS, S_k + [current]) - coverage)

            Q = sorted(Q, key=lambda x: x[1], reverse=True)

            found = Q[0][0] == current

        S_k.append(Q[0][0])
        coverage += Q[0][1]
        Q = Q[1:]

    return S_k


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
            RRS.append(imp.get_a_unused_rrset())

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
        RRS.append(imp.get_a_unused_rrset())

    return RRS


def IMM(imp):
    global IMM_para_l
    IMM_para_l0 = IMM_para_l

    IMM_para_l = IMM_para_l * (1 + np.log(2) / np.log(imp.V))
    RRS = IMM_sampling(imp)
    S_k = IMM_node_selection(RRS, imp.k, imp.V)

    IMM_para_l = IMM_para_l0

    return S_k, RRS


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

