"""
NSGA-II and CELF on influence maximization problem (IM)

"""
# -*- coding: utf-8 -*-
import copy
import datetime
import itertools
import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import pylab
import inspyred
import numpy as np
import pandas as pd
from time import time
from inspyred.ec import emo
from inspyred.ec.variators import crossover


def IC(G, S, p=0.5, mc=10000):
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
            temp = G.loc[G['source'].isin(new_active)]

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


def CELF(G, p=0.5, mc=10000):
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
    candidates = np.unique(G['source'])
    marg_gain = [IC(G, [c], p=p, mc=mc) / cost for c in candidates]
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
            Q[0] = (current, IC(G, S + [current], p=p, mc=mc) - spread)

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


def get_influence(RRS, seeds):
    """
        Inputs: RRS: random RR set
                seeds: a seed set of nodes, represented by bitmap
                num_nodes: number of the nodes in the network
        Return: a value represents the estimation of influence of the seeds
    """
    count = 0
    for seed_set in RRS:
        for node in seed_set:
            if seeds[node - 1] > 0.5:
                count = count + 1
                break
    return count / len(RRS) * len(seeds)


def get_random_RRS(G, p=0.5):
    """
    Inputs: G:  Ex2 dataframe of directed edges. Columns: ['source','target']
    Return: A random reverse reachable set expressed as a list of nodes
    """

    # Step 1. Select random source node
    source = np.random.choice(np.unique(G['source']))

    # Step 2. Get an instance of g
    g = G.copy().loc[np.random.uniform(0, 1, G.shape[0]) < p]

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


def get_RRS(use_file, num_rrset, G, name='', p=0.5):
    RRS = []
    if not use_file:
        RRS = []
        print("creating RRS")
        print("\r{:3}%".format(0), end="")
        for _i in range(0, num_rrset):
            r = get_random_RRS(G=G, p=p)
            RRS.append(r)
            print("\r{:3}%".format((_i + 1) / num_rrset * 100), end="")
        print("\n")
        try:
            np.savetxt(result_path + name + "/RRS-out.txt", RRS, delimiter=", ", fmt="% s")
        except FileNotFoundError:
            os.makedirs(result_path + name)
            np.savetxt(result_path + name + "/RRS-out.txt", RRS, delimiter=", ", fmt="% s")
    else:
        RRS_file = open(result_path + name + "/RRS-out.txt")
        for rf in RRS_file.readlines():
            rf = rf.strip('[]\n')
            RRS.append(list(map(int, rf.split(","))))
        print("read RRS from file, size : " + str(len(RRS)))

    return RRS


# -------------
# these are for the inspyred framework
# -------------


def my_generator(random, args):
    num_users = args.get('num_users')
    return [random.random() for _ in range(num_users)]


def my_evaluator(candidates, args):
    fitness = []
    RRS = args.get('RRS')
    for cs in candidates:
        rec_cost = 0
        for c in cs:
            if c > 0.5:
                rec_cost = rec_cost + 1
        influence_spread = get_influence(RRS, cs)
        # fitness.append([-influence_spread, rec_cost])
        fitness.append(emo.Pareto([-influence_spread, rec_cost]))
    return fitness


def my_bound(candidate, args):
    for _i, c in enumerate(candidate):
        candidate[_i] = max(min(c, 1), 0)
    return candidate


my_bound.lower_bound = itertools.repeat(0)
my_bound.upper_bound = itertools.repeat(1)


def my_observer(population, num_generations, num_evaluations, args):
    print("\rgen : {:3}".format(num_generations), end="")
    if args.get('max_generations') == num_generations:
        print(' ...done')


@crossover
def my_cross(random, mom, dad, args):
    blx_points = args.setdefault('blx_points', None)
    crossover_rate = args.setdefault('crossover_rate', 1.0)
    bounder = args['_ec'].bounder
    children = []
    if random.random() < crossover_rate:
        bro = copy.copy(dad)
        sis = copy.copy(mom)
        if blx_points is None:
            blx_points = list(range(min(len(bro), len(sis))))
        for i in blx_points:
            avg = (mom[i] + dad[i]) / 2
            bro[i] = avg + random.random() * 0.3
            sis[i] = avg - random.random() * 0.3
        bro = bounder(bro, args)
        sis = bounder(sis, args)
        children.append(bro)
        children.append(sis)
    else:
        children.append(mom)
        children.append(dad)
    return children


def EC_optimization(RRS, _user_num, pop_size=100, max_generation=500, prng=None):
    """
    my modified nsga-ii
    :param RRS:
    :param _user_num: vertex number
    :param pop_size:
    :param max_generation:
    :param prng: I don't know what it is, please check the inspyred doc

    :return: values of final archive set [[influence], [cost]]
    """

    # TODO
    # Disease propagation probability should be part of the network

    start_time = time()
    # start optimization
    if prng is None:
        prng = random.Random()
        prng.seed(time())

    ea = inspyred.ec.emo.NSGA2(prng)
    # blend arithmetic think deep in binary
    ea.variator = [
        my_cross,
        inspyred.ec.variators.gaussian_mutation
    ]
    ea.terminator = inspyred.ec.terminators.generation_termination
    ea.observer = my_observer

    ea.evolve(
        evaluator=my_evaluator,
        generator=my_generator,
        bounder=my_bound,
        pop_size=pop_size,
        maximize=False,
        max_generations=max_generation,
        num_users=_user_num,
        RRS=RRS,
        blx_alpha=1.0)
    print(str(time() - start_time) + 's\n')

    _x = []
    _y = []
    for agent in ea.archive:
        _x.append(-agent.fitness[0])
        _y.append(agent.fitness[1])

    return [_x, _y]


def width_of_RR_set(G, R):
    # TODO need to test it
    # TODO maybe a table of income degree
    vc = G['target'].value_counts()
    width = 0
    for node in R:
        try:
            width = width + vc[node]
        except KeyError:
            width = width

    return width


_l = 1


def KPT_estimation(G, k, n, m, p=0.5):
    for i in range(1, int(np.log2(n) - 1)):
        ci = 6 * _l * np.log(n) + 6 * np.log(np.log2(n)) * np.exp2(i)
        _sum = 0
        for j in range(1, int(ci)):
            R = get_random_RRS(G, p)
            kR = 1 - (1 - width_of_RR_set(G, R) / m) ** k
            _sum = _sum + kR
        if _sum / ci > 1 / np.exp2(i):
            return n * _sum / (2 * ci)
    return 1


def node_selection(G, k, theta, p=0.5):
    # Generate theta random RR sets and insert them into set_R
    set_R = get_RRS(False, int(theta), G, p=p, name=_dataset_name)
    # for _ in range(0, int(theta)):
    #     set_R.append(get_random_RRS(G, p=p))

    # S_k as the solution, and I need another list to record the trace of S_k
    S_k = []
    trace = []

    for j in range(1, k + 1):
        # identify the node that covers the most RR sets in set_R
        flat_list = [item for sublist in set_R for item in sublist]
        most_common = Counter(flat_list).most_common()
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


def TIM(G, k, n, m, p=0.5):
    epsilon = 0.3
    KPT_star = KPT_estimation(G, k, n, m, p=p)
    _lambda = (8 + 2 * epsilon) * n * (_l * np.log(n) + np.log(float(np.math.comb(n, k))) + np.log(2)) * epsilon ** (-2)
    theta = _lambda / KPT_star

    print("theta : " + str(theta))
    influence = []
    cost = []
    trace = node_selection(G, k, theta, p=p)

    print("start IC : ")
    step = int(len(trace) / 10)
    pos = 0
    while pos <= len(trace):
        print("pos : " + str(pos))
        influence.append(IC(G, trace[pos], p=0.5, mc=10000))
        cost.append(len(trace[pos]))

        pos = pos + step

    return [influence, cost]


def show_result(results, labels, name):
    color = ['b', 'r', 'g']

    fig, ax = plt.subplots()
    ax.set_xlabel('Influence spread')
    ax.set_ylabel('Recruitment costs')
    for _i in range(len(results)):
        ax.scatter(results[_i][0],
                   results[_i][1],
                   s=3,
                   c=color[_i],
                   alpha=0.8,
                   label=labels[_i])
    ax.legend()
    ax.grid(True)
    plt.title(name)
    # plt.show()
    file_name = result_path + name + '/result' + \
                    str(datetime.datetime.now().minute) + str(datetime.datetime.now().hour) + '.pdf'
    try:
        pylab.savefig(file_name, format='pdf')
    except FileNotFoundError:
        os.makedirs(result_path + name)
        pylab.savefig(file_name, format='pdf')


def main():
    # read config file
    my_datasets = pd.read_csv(datasets_path + config_file, delimiter=',', index_col=False, skipinitialspace=True)
    print("read config file")
    print(my_datasets[['name', 'activate', 'V', 'E', "use_file"]])
    print("==========")

    for idx in my_datasets.index:

        # check dataset
        if not my_datasets['activate'][idx]:
            continue
        if my_datasets['weighted'][idx]:
            print("can not process weight network")
            continue

        dataset_name = str(my_datasets['name'][idx])
        #
        global _dataset_name
        _dataset_name = dataset_name

        num_node = int(my_datasets['V'][idx])
        num_edge = int(my_datasets['E'][idx])
        print(dataset_name + "\nnode_num is : " + str(num_node))

        # create graph
        idx_G = ['source', 'target']
        G = pd.read_csv(
            datasets_path + dataset_name + '.mtx',
            delimiter=" ",
            index_col=False,
            names=idx_G,
            skiprows=2
        )

        # if network is directed
        if not my_datasets["directed"][idx]:
            print("this data is undirected")
            G_copy = G.copy(deep=True)
            G_copy[['target', 'source']] = G_copy[['source', 'target']]
            G_copy.columns = ['source', 'target']
            G = pd.concat([G, G_copy], ignore_index=True)

            num_edge = num_edge * 2

        # run algorithms
        k = int(num_node * 0.7)
        result_TIM = TIM(G, k, num_node, num_edge, p=0.5)

        RRS = get_RRS(True, int(my_datasets['sketch_num'][idx]), G, name=dataset_name, p=0.5)
        result_EC = EC_optimization(
            RRS,
            num_node,
            pop_size=int(my_datasets['pop_size'][idx]),
            max_generation=int(my_datasets['max_generation'][idx])
        )

        # result_celf = CELF(G, p=0.5, mc=10000)
        # visualization
        # show_result([result_EC, result_celf],
        #             ["NSGA-II", "CELF"], name=dataset_name)
        # show_result([result_EC],
        #             ["NSGA-II"], name=dataset_name)
        #
        # show_result([result_EC, result_TIM],
        #             ["NSGA-II", "TIM"], name=dataset_name)

        show_result([result_TIM, result_EC],
                    ["TIM", "NSGA-II"], name=dataset_name)
        print("\n===============")


# global var
result_path = "./result/"
datasets_path = "./datasets/"
config_file = "config.csv"
_dataset_name = ""
main()
