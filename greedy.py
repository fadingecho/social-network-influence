import copy
from collections import Counter
from time import time

import numpy as np


def CELF(IM_dataset, p=0.5, mc=10000):
    """
    Inputs: G:  Ex2 dataframe of directed edges. Columns: ['source','target']
            p:  Disease propagation probability
            mc: Number of Monte-Carlo simulations
    Return: greedy_trace: list[cost] = influence
    """

    # --------------------
    # Find the first node with greedy algorithm
    # --------------------

    # Compute marginal gain for each node
    # cost is equal
    print("start celf_mc")
    start_time = time()
    cost = 1
    candidates = np.unique(IM_dataset.G['source'])
    marg_gain = [IM_dataset.IC([c], p=p, mc=mc) / cost for c in candidates]
    # Create the sorted list of nodes and their marginal gain
    Q = sorted(zip(candidates, marg_gain), key=lambda x: x[1], reverse=True)

    # Select the first node and remove from candidate list
    S, spread, Q = [Q[0][0]], Q[0][1], Q[1:]
    greedy_trace = [spread]

    # --------------------
    # Find the next k-1 nodes using the CELF list-sorting procedure
    # --------------------

    print("\ri : {:3}".format(0), end="")
    for _i in range(len(Q)):
        print("\ri : {:3}".format(_i), end="")
        found = False

        while not found:
            # Recalculate spread of top node
            current = Q[0][0]

            # Evaluate the spread function and store the marginal gain in the list
            Q[0] = (current, IM_dataset.IC(S + [current], p=p, mc=mc) - spread)

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

    print(" ...done")
    print(str(time() - start_time) + 's')

    return [greedy_trace, [_i + 1 for _i in range(len(greedy_trace))]]


def node_selection(IM_dataset, p=0.5):

    # Generate theta random RR sets and insert them into set_R
    set_R = IM_dataset.RRS

    # code below is wrong
    # for _ in range(0, int(theta)):
    #     set_R.append(get_random_RRS(G, p=p))

    # S_k as the solution, and I need another list to record the trace of S_k

    S_k = []
    trace = []

    for j in range(1, IM_dataset.k + 1):
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

    return trace


def TIM(IM_dataset, p=0.5):
    # estimation part is regarded as part of the problem, check it in the class IMP.py
    influence = []
    cost = []
    trace = node_selection(IM_dataset, p=p)

    print("start IC : ")
    step = int(len(trace) / 10)
    pos = 0
    while pos - 1 < len(trace):
        print("pos : " + str(pos))
        try:
            influence.append(IM_dataset.IC(trace[pos], p=0.5, mc=10000))
        except IndexError:
            print(str(pos) + " " + str(len(trace)))
        cost.append(len(trace[pos]))

        pos = pos + step

    return [influence, cost]

