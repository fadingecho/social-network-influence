# -*- coding: utf-8 -*-
import itertools
import random
import re
import pylab
import inspyred
import scipy as sp
import numpy as np
import pandas as pd
from collections import Counter
from time import time

import scipy.io


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


def ris(_G, k, p=0.5, mc=1000):
    print("ris start")
    print("\r{:3}%".format(0), end="")
    """
    Inputs: G:  Ex2 dataframe of directed edges. Columns: ['source','target']
            k:  Size of seed set
            p:  Disease propagation probability
            mc: Number of RRSs to generate
    Return: A seed set of nodes as an approximate solution to the IM problem
    """

    # Step 1. Generate the collection of random RRSs
    start_time = time()
    R = []
    for _i in range(mc):
        R.append(get_random_RRS(_G=_G))
        print("\r{:3}%".format((_i + 1) / mc * 100), end="")
    print("\n")

    # Step 2. Choose nodes that appear most often (maximum coverage greedy algorithm)
    SEED, timelapse = [], []
    for _i in range(k):
        # Find node that occurs most often in R and add to seed set
        flat_list = [item for sublist in R for item in sublist]
        seed = Counter(flat_list).most_common()[0][0]
        SEED.append(seed)

        # Remove RRSs containing last chosen seed
        R = [rrs for rrs in R if seed not in rrs]

        # Record Time
        timelapse.append(time() - start_time)
        if not R:
            break

    return sorted(SEED), timelapse


def get_random_RRS(_G, sketches):
    """
    Inputs: G:  Ex2 dataframe of directed edges. Columns: ['source','target']
    Return: A random reverse reachable set expressed as a list of nodes
    """

    # Step 1. Select random source node
    source = np.random.choice(np.unique(_G['source']))

    # Step 2. Get an instance of g
    g = sketches[np.random.randint(0, len(sketches))]

    # Step 3. Construct reverse reachable set of the random source node
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


def _my_generator(random, args):
    num_users = args.get('num_users')
    return [random.random() for _ in range(num_users)]


def _my_evaluator(candidates, args):
    values = []
    RRS = args.get('RRS')
    num_users = args.get('num_users')
    for cs in candidates:
        rec_cost = 0
        for c in cs:
            if c > 0.5:
                rec_cost = rec_cost + 1
        influence_spread = get_influence(RRS, cs, num_users)
        values.append([-influence_spread, rec_cost])
    return values


def _my_bound(candidate, args):
    for _i, c in enumerate(candidate):
        candidate[_i] = max(min(c, 1), 0)
    return candidate


_my_bound.lower_bound = itertools.repeat(0)
_my_bound.upper_bound = itertools.repeat(1)


def optimization(_G, _user_num, prng=None, display=False):
    # TODO
    # timer in bar
    # parse matrix
    # use file to skip initialization

    # create sketch
    sketches = []
    len_sketch = 1000
    p = 0.3
    weight_range = max(_G['weight']) - min(_G['weight']) + 1
    print("creating sketch")
    print("\r{:3}%".format(0), end="")
    for _i in range(0, len_sketch):
        # g = pd.DataFrame(columns=['source', 'target'], dtype=int)
        # for row in _G.itertuples():
        #     b = getattr(row, 'weight') > weight_range * np.random.rand()
        #     if b:
        #         g = g.append({"source": getattr(row, 'source'), "target": getattr(row, 'target')}, ignore_index=True)
        g = _G.copy().loc[np.random.uniform(0, 1, _G.shape[0]) < p]
        sketches.append(g)
        print("\r{:3}%".format((_i + 1) / len_sketch * 100), end="")
    print("\n")
    np.savetxt("sketches-out.txt", sketches, delimiter=", ", fmt="% s")

    # create RRS according to sketches
    RRS = []
    len_RRS = 50000
    print("creating RRS")
    print("\r{:3}%".format(0), end="")
    for _i in range(0, len_RRS):
        r = get_random_RRS(_G=_G, sketches=sketches)
        RRS.append(r)
        print("\r{:3}%".format((_i + 1) / len_RRS * 100), end="")
    print("\n")
    np.savetxt("RRS-out.csv", RRS, delimiter=", ", fmt="% s")

    #
    if prng is None:
        prng = random.Random()
        prng.seed(time())

    ea = inspyred.ec.emo.NSGA2(prng)
    ea.variator = [inspyred.ec.variators.blend_crossover,
                   inspyred.ec.variators.gaussian_mutation]
    ea.terminator = inspyred.ec.terminators.generation_termination
    final_pop = ea.evolve(
        evaluator=_my_evaluator,
        generator=_my_generator,
        bounder=_my_bound,
        pop_size=100,
        maximize=False,
        max_generations=100,
        num_users=_user_num,
        RRS=RRS)

    #
    if display:
        final_arc = ea.archive
        np.savetxt("pop-out.csv", ea.population, delimiter=", ", fmt="% s")
        x = []
        y = []
        for agent in final_arc:
            x.append(-agent.fitness[0])
            y.append(agent.fitness[1])
        pylab.xlabel('influence spread')
        pylab.ylabel('recruit cost')
        pylab.scatter(x, y, color='b')
        pylab.savefig('{0} ({1}).pdf'.format(ea.__class__.__name__, 'IM'),
                      format='pdf')
        pylab.show()
    return ea


if __name__ == "__main__":
    path = 'soc-douban.mtx'
    f = open(path, 'r')
    f.readline()
    user_num = int(f.readline().split(" ")[1])
    print("user_num is : " + str(user_num))
    f.close()

    idx_G = ['source', 'target', 'weight']
    G = pd.read_csv(
        path,
        delimiter=" ",
        index_col=False,
        names=idx_G,
        skiprows=2)

    sym = False
    if sym:
        G_copy = G.copy(deep=True)
        G_copy[['target', 'source']] = G_copy[['source', 'target']]
        G_copy.columns = ['source', 'target']
        G = pd.concat([G, G_copy], ignore_index=True)
    optimization(G, user_num, display=True)

    # test out files
    # f = open("pop-out.csv", 'r')
    # DNAs = []
    # seeds = []
    # values = []
    # lines = f.readlines()
    # for line in lines:
    #     items = re.split(r"[\[\]]", line)
    #     DNA = list(map(float, items[1].split(",")))
    #     DNAs.append(DNA)
    #     seed = []
    #     for i in range(0, len(DNA)):
    #         if DNA[i] > 0.5:
    #             seed.append(i + 1)
    #     seeds.append(seed)
    #     value = items[3].split(",")
    #     values.append([float(value[0]), int(value[1])])
    #
    # x = []
    # y = []
    # for value in values:
    #     x.append(-value[0])
    #     y.append(value[1])
    # pylab.xlabel('influence spread')
    # pylab.ylabel('recruit cost')
    # pylab.scatter(x, y, color='b')
    # pylab.savefig('{0} ({1}).pdf'.format("pop", "test"),
    #               format='pdf')
    # pylab.show()
    # f.close()
