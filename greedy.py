import copy
from collections import Counter
from time import time
import numpy as np
import _utils


def CELF(IM_dataset):
    """
    Inputs: G:  Ex2 dataframe of directed edges. Columns: ['source','target']
            p:  Disease propagation probability
    Return: greedy_trace: list[cost] = influence
    """

    # --------------------
    # Find the first node with greedy algorithm
    # --------------------
    start_time = time()
    # cost is equal
    cost = 1
    # Compute marginal gain for each node
    candidates = np.unique(IM_dataset.G['source'])
    marg_gain = [IM_dataset.IC([c]) / cost for c in candidates]

    # Create the sorted list of nodes and their marginal gain
    Q = sorted(zip(candidates, marg_gain), key=lambda x: x[1], reverse=True)

    # Select the first node and remove from candidate list
    S, spread, Q = [Q[0][0]], Q[0][1], Q[1:]
    greedy_trace = [spread]

    # --------------------
    # Find the next k-1 nodes using the CELF list-sorting procedure
    # --------------------

    k = IM_dataset.k
    _utils.show_process_bar("CELF", 0, k)
    for _i in range(k):
        _utils.show_process_bar("CELF", _i, k)
        found = False

        while not found:
            # Recalculate spread of top node
            current = Q[0][0]

            # Evaluate the spread function and store the marginal gain in the list
            Q[0] = (current, IM_dataset.IC(S + [current]) - spread)

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


def node_selection(IM_dataset):
    """
        Inputs: IM_dataset:  provide estimated RRS, check IM_problem.py
        Return: greedy_trace: list[cost] = influence
    """

    # Generate theta random RR sets and insert them into set_R
    set_R = IM_dataset.RRS

    # S_k is the final solution
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


def TIM(imp):
    """
        Inputs: IM_dataset:  provide estimated RRS, check IM_problem.py
                p:  Disease propagation probability
        Return: greedy trace
    """
    # estimation part is regarded as part of the problem model, check it in the class IM_problem.py
    trace = node_selection(imp)

    return trace


def IC_evaluate(trace, imp):
    influence = []
    cost = []
    # to save time, I calculate the spread of Monte-Carlo with a fixed step length
    len_trace = len(trace)
    # step = int(len_trace / 15)
    step = 1
    if step == 0:
        step = 1
    pos = 0

    while pos < len_trace:
        _utils.show_process_bar("IC evaluating", pos + 1, len_trace)

        influence.append(imp.IC(trace[pos]))
        cost.append(len(trace[pos]))
        pos = pos + step
    _utils.show_process_bar("IC evaluating", len_trace, len_trace)
    if pos < len_trace - 1:
        influence.append(imp.IC(trace[len_trace - 1]))
        cost.append(len(trace[len_trace - 1]))
    _utils.process_end("")

    return [influence, cost]
