# -*- coding: utf-8 -*-
import itertools
import random
import re
import pylab
import inspyred
import numpy as np
import pandas as pd
from collections import Counter
from time import time

import scipy.io
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
        print('\n')


def optimization(_G, _user_num, prng=None, display=False):
    # TODO
    # timer in bar
    # parse matrix
    # use file to skip initialization

    use_file = True
    RRS = []
    if not use_file:
        # create sketch
        sketches = []
        len_sketch = 10000
        weighted = False
        if weighted:
            weight_range = max(_G['weight']) - min(_G['weight']) + 1
        p = 0.5
        print("creating sketch")
        print("\r{:3}%".format(0), end="")
        for _i in range(0, len_sketch):
            if weighted:
                g = pd.DataFrame(columns=['source', 'target'], dtype=int)
                for row in _G.itertuples():
                    # sample edges based on weight
                    # ?
                    if p ** getattr(row, 'weight') < np.random.uniform(0, 1):
                        g = g.append({"source": getattr(row, 'source'), "target": getattr(row, 'target')},
                                     ignore_index=True)
            else:
                g = _G.copy().loc[np.random.uniform(0, 1, _G.shape[0]) < p]
            sketches.append(g)
            print("\r{:3}%".format((_i + 1) / len_sketch * 100), end="")
        print("\n")
        # print(sketches)
        np.savetxt("./result/sketches-out.txt", sketches, delimiter=", ", fmt="% s")

        # create RRS according to sketches
        RRS = []
        print("creating RRS")
        print("\r{:3}%".format(0), end="")
        for _i in range(0, len_sketch):
            for _ in range(0, 10):
                r = get_random_RRS(_G=_G, sketch=sketches[_i])
                RRS.append(r)
            print("\r{:3}%".format((_i + 1) / len_sketch * 100), end="")
        print("\n")
        np.savetxt("./result/RRS-out.txt", RRS, delimiter=", ", fmt="% s")
    else:
        RRS_file = open("./result/RRS-out.txt")
        for rf in RRS_file.readlines():
            rf = rf.strip('[]\n')
            RRS.append(list(map(int, rf.split(","))))
        print("read RRS from file")
    #
    if prng is None:
        prng = random.Random()
        prng.seed(time())

    ea = inspyred.ec.emo.NSGA2(prng)
    ea.variator = [inspyred.ec.variators.blend_crossover,
                   inspyred.ec.variators.gaussian_mutation]
    ea.terminator = inspyred.ec.terminators.generation_termination
    ea.observer = my_observer
    final_pop = ea.evolve(
        evaluator=_my_evaluator,
        generator=_my_generator,
        bounder=_my_bound,
        pop_size=100,
        maximize=False,
        max_generations=10,
        num_users=_user_num,
        RRS=RRS)

    #
    if display:
        final_arc = ea.archive
        np.savetxt("./result/pop.txt", ea.population, delimiter=", ", fmt="% s")
        np.savetxt("./result/arc.txt", ea.archive, delimiter=", ", fmt="% s")
        _x = []
        _y = []
        for agent in final_arc:
            _x.append(-agent.fitness[0])
            _y.append(agent.fitness[1])
        pylab.xlabel('Influence spread')
        pylab.ylabel('Recruitment costs')
        pylab.scatter(_x, _y, color='b')
        pylab.savefig('./result/{0} {1}.pdf'.format(ea.__class__.__name__, 'IM'),
                      format='pdf')
        pylab.show()
    return ea


if __name__ == "__main__":
    path = 'soc-wiki-Vote.mtx'
    f = open(path, 'r')
    f.readline()
    user_num = int(f.readline().split(" ")[0])
    print("user_num is : " + str(user_num))
    f.close()

    idx_G = ['source', 'target']
    G = pd.read_csv(
        path,
        delimiter=" ",
        index_col=False,
        names=idx_G,
        skiprows=2)

    directed = False
    if directed:
        G_copy = G.copy(deep=True)
        G_copy[['target', 'source']] = G_copy[['source', 'target']]
        G_copy.columns = ['source', 'target']
        G = pd.concat([G, G_copy], ignore_index=True)
    optimization(G, user_num, display=True)

    # test out files
    f = open("./result/pop.txt", 'r')
    # DNAs = []
    # seeds = []
    values = []
    lines = f.readlines()
    for line in lines:
        items = re.split(r"[\[\]]", line)
        # DNA = list(map(float, items[1].split(",")))
        # DNAs.append(DNA)
        # seed = []
        # for i in range(0, len(DNA)):
        #     if DNA[i] > 0.5:
        #         seed.append(i + 1)
        # seeds.append(seed)
        value = items[3].split(",")
        values.append([float(value[0]), int(value[1])])
    #
    x = []
    y = []
    for value in values:
        x.append(-value[0])
        y.append(value[1])
    pylab.xlabel('influence spread')
    pylab.ylabel('recruit cost')
    pylab.scatter(x, y, color='b')
    pylab.savefig('./result/{0}.pdf'.format("pop"),
                  format='pdf')
    pylab.show()
    f.close()
