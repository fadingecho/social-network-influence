"""
NSGA-II and CELF on influence maximization problem (IM)

"""
# -*- coding: utf-8 -*-
import itertools
import os
import random
import traceback
import matplotlib.pyplot as plt
import pylab
import inspyred
import numpy as np
import pandas as pd
from time import time
from inspyred.ec import emo


def IC(_G, S, p=0.5, mc=1000):
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
            temp = _G.loc[_G['source'].isin(new_active)]

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


def celf(_G, p=0.6, mc=1000):
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
    print("start celf")
    start_time = time()
    cost = 1
    candidates = np.unique(G['source'])
    marg_gain = [IC(G, [c], p=p, mc=mc) / cost for c in candidates]
    # Create the sorted list of nodes and their marginal gain
    Q = sorted(zip(candidates, marg_gain), key=lambda x: x[1], reverse=True)

    # Select the first node and remove from candidate list
    S, spread, Q = [Q[0][0]], Q[0][1], Q[1:]
    greedy_trace = [0, spread]

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

    return [greedy_trace, [_i for _i in range(len(greedy_trace))]]


def get_random_RRS(_G, sketch):
    """
    Inputs: G:  Ex2 dataframe of directed edges. Columns: ['source','target']
    Return: A random reverse reachable set expressed as a list of nodes
    """

    # Step 1. Select random source node
    source = np.random.choice(np.unique(_G['source']))

    # Step 2. Get an instance of g

    # Step 3. Construct reverse reachable set of the random source node
    new_nodes, RRS0 = [source], [source]
    while new_nodes:
        # Limit to edges that flow into the source node
        temp = sketch.loc[sketch['target'].isin(new_nodes)]

        # Extract the nodes flowing into the source node
        temp = temp['source'].tolist()

        # Add new set of in-neighbors to the RRS
        RRS = list(set(RRS0 + temp))

        # Find what new nodes were added
        new_nodes = list(set(RRS) - set(RRS0))

        # Reset loop variables
        RRS0 = RRS[:]

    return (RRS)


def get_influence(RRS, seeds, num_nodes):
    """
        Inputs: RRS: random RR set
                seeds: a seed set of nodes
                num_nodes: number of the nodes in the network
        Return: a value represents the estimation of influence of the seeds
    """
    count = 0
    for seed_set in RRS:
        for node in seed_set:
            if seeds[node - 1] > 0.5:
                count = count + 1
                break
    return count / len(RRS) * num_nodes


# -------------
# these are for the inspyred framework
# -------------


def _my_generator(random, args):
    num_users = args.get('num_users')
    return [random.random() for _ in range(num_users)]


def _my_evaluator(candidates, args):
    fitness = []
    RRS = args.get('RRS')
    num_users = args.get('num_users')
    for cs in candidates:
        rec_cost = 0
        for c in cs:
            if c > 0.5:
                rec_cost = rec_cost + 1
        influence_spread = get_influence(RRS, cs, num_users)
        # fitness.append([-influence_spread, rec_cost])
        fitness.append(emo.Pareto([-influence_spread, rec_cost]))
    return fitness


def _my_bound(candidate, args):
    for _i, c in enumerate(candidate):
        candidate[_i] = max(min(c, 1), 0)
    return candidate


_my_bound.lower_bound = itertools.repeat(0)
_my_bound.upper_bound = itertools.repeat(1)


def my_observer(population, num_generations, num_evaluations, args):
    print("\rgen : {:3}".format(num_generations), end="")
    if args.get('max_generations') == num_generations:
        print(' ...done')


def EC_optimization(_G, _user_num, pop_size=100, max_generation=500, sketch_num=10000, prng=None):
    """
    my modified nsga-ii
    :param _G: network
    :param _user_num: vertex number
    :param pop_size:
    :param max_generation:
    :param sketch_num: number of sketch for rr sets
    :param prng: I don't know what it is
    :return: values of final archive set [[influence], [cost]]
    """

    # TODO
    # use file

    start_time = time()

    # step 1 : get rr sets
    # if we've already got the rr sets for this network
    use_file = False
    RRS = []
    if not use_file:
        # create sketch
        sketches = []
        p = 0.5
        print("creating sketch")
        print("\r{:3}%".format(0), end="")
        for _i in range(0, sketch_num):
            g = _G.copy().loc[np.random.uniform(0, 1, _G.shape[0]) < p]
            sketches.append(g)
            print("\r{:3}%".format((_i + 1) / sketch_num * 100), end="")
        print("\n")

        #
        # create RRS according to sketches
        RRS = []
        print("creating RRS")
        print("\r{:3}%".format(0), end="")
        for _i in range(0, sketch_num):
            for _ in range(0, 10):
                r = get_random_RRS(_G=_G, sketch=sketches[_i])
                RRS.append(r)
            print("\r{:3}%".format((_i + 1) / sketch_num * 100), end="")
        print("\n")
        try:
            np.savetxt(result_path + dataset_name + "/RRS-out.txt", RRS, delimiter=", ", fmt="% s")
        except FileNotFoundError:
            os.makedirs(result_path + dataset_name)
            np.savetxt(result_path + dataset_name + "/RRS-out.txt", RRS, delimiter=", ", fmt="% s")
    else:
        try:
            RRS_file = open(result_path + dataset_name + "RRS-out.txt")
        except FileNotFoundError:
            print("RRS-out.txt lost!")
            traceback.print_exc()

        for rf in RRS_file.readlines():
            rf = rf.strip('[]\n')
            RRS.append(list(map(int, rf.split(","))))
        print("read RRS from file")

    # start optimization
    if prng is None:
        prng = random.Random()
        prng.seed(time())

    ea = inspyred.ec.emo.NSGA2(prng)
    ea.variator = [
        inspyred.ec.variators.blend_crossover,
        inspyred.ec.variators.gaussian_mutation
    ]
    ea.terminator = inspyred.ec.terminators.generation_termination
    ea.observer = my_observer

    final_pop = ea.evolve(
        evaluator=_my_evaluator,
        generator=_my_generator,
        bounder=_my_bound,
        pop_size=pop_size,
        maximize=False,
        max_generations=max_generation,
        num_users=_user_num,
        RRS=RRS)
    print(str(time() - start_time) + 's\n')

    # process the result
    try:
        np.savetxt(result_path + dataset_name + "/pop.txt", ea.population, delimiter=", ", fmt="% s")
        np.savetxt(result_path + dataset_name + "/arc.txt", ea.archive, delimiter=", ", fmt="% s")
    except FileNotFoundError:
        os.makedirs(result_path + dataset_name)
        np.savetxt(result_path + dataset_name + "/pop.txt", ea.population, delimiter=", ", fmt="% s")
        np.savetxt(result_path + dataset_name + "/arc.txt", ea.archive, delimiter=", ", fmt="% s")

    _x = []
    _y = []
    for agent in ea.archive:
        _x.append(-agent.fitness[0])
        _y.append(agent.fitness[1])

    return [_x, _y]


def show_result(results, labels):
    color = ['b', 'r']

    fig, ax = plt.subplots()
    ax.set_xlabel('Influence spread')
    ax.set_ylabel('Recruitment costs')
    for _i in range(len(results)):
        ax.scatter(results[_i][0],
                   results[_i][1],
                   s=1,
                   c=color[_i],
                   alpha=0.5,
                   label=labels[_i])
    ax.legend()
    ax.grid(True)

    try:
        pylab.savefig(result_path + dataset_name + '/result.pdf', format='pdf')
    except FileNotFoundError:
        os.makedirs(result_path + dataset_name)
        pylab.savefig(result_path + dataset_name + '/result.pdf', format='pdf')


# global var
result_path = "./result/"
datasets_path = "./datasets/"
config_file = "config.csv"
dataset_name = ""

if __name__ == "__main__":
    # read config file
    my_datasets = pd.read_csv(datasets_path + config_file, delimiter=',', index_col=False)
    print("read config file")
    print(my_datasets[['name', 'activate', 'V', 'E']])
    print("==========")

    for idx in my_datasets.index:

        # check dataset
        if not my_datasets['activate'][idx]:
            continue
        if my_datasets['weighted'][idx]:
            print("can not process weight network")
            continue

        dataset_name = str(my_datasets['name'][idx])
        user_num = int(my_datasets['V'][idx])
        print(dataset_name + "\nuser_num is : " + str(user_num))

        # create graph
        idx_G = ['source', 'target']
        G = pd.read_csv(
            datasets_path + dataset_name + '.mtx',
            delimiter=" ",
            index_col=False,
            names=idx_G,
            skiprows=2)

        # if network is directed
        if not my_datasets["directed"][idx]:
            print("this data is undirected")
            G_copy = G.copy(deep=True)
            G_copy[['target', 'source']] = G_copy[['source', 'target']]
            G_copy.columns = ['source', 'target']
            G = pd.concat([G, G_copy], ignore_index=True)

        # run the algorithms
        result_EC = EC_optimization(
            G,
            user_num,
            sketch_num=int(my_datasets['sketch_num'][idx]),
            pop_size=int(my_datasets['pop_size'][idx]),
            max_generation=int(my_datasets['max_generation'][idx])
        )
        result_celf = celf(G, mc=int(my_datasets['mc'][idx]))

        # visualization
        show_result([result_EC, result_celf],
                    ["NSGA-II", "CELF"])
        print("\n===============")
