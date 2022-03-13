import copy
import itertools
import random
from time import time

import inspyred
from inspyred.ec import emo
from inspyred.ec.generators import diversify
from inspyred.ec.variators import crossover

import utils


@diversify
def generator_function(random, args):
    return random.sample(list(range(1, args.get('num_users'))), k=random.randint(1, args.get('k')))


def get_cost(seeds):
    return len(seeds)


def get_influence(RRS, seeds, args):
    """
        Inputs: RRS: random RR set
                seeds: a seed set of nodes, represented by a node set
                num_nodes: number of the nodes in the network
        Return: a value represents the estimation of influence of the seeds
    """

    count = 0
    # count the frequency of intersection between the seed set and every rr set
    for r in RRS:
        for s in seeds:
            if s in r:
                count = count + 1
                break
    return count / len(RRS) * args.get('num_users')


def my_evaluator(candidates, args):
    fitness = []
    RRS = args.get('RRS')
    for cs in candidates:
        rec_cost = get_cost(cs)
        influence_spread = get_influence(RRS, cs, args)

        fitness.append(emo.Pareto([-influence_spread, rec_cost]))
    return fitness


def my_observer(population, num_generations, num_evaluations, args):
    max_gen = args.get('max_generations')
    utils.show_process_bar("gen :", num_generations, max_gen)


# TODO sample from flat RRS
@crossover
def my_cross(random, mom, dad, args):
    alpha = 0.3
    alpha_p = 0.5
    beta = 0.3
    beta_p = 0.5

    crossover_rate = args.setdefault('crossover_rate', 1.0)
    children = []
    if random.random() < crossover_rate:
        bro = copy.copy(dad)
        sis = copy.copy(mom)

        # add node
        dad_attitude = random.sample(dad, int(alpha * len(dad)))
        for d in dad_attitude:
            if d not in mom:
                if random.random() > alpha_p:
                    sis.append(d)
        mom_attitude = random.sample(mom, int(alpha * len(mom)))
        for m in mom_attitude:
            if m not in dad:
                if random.random() > alpha_p:
                    bro.append(m)

        # remove node
        dad_attitude = random.sample(mom, int(beta * len(mom)))
        for d in dad_attitude:
            if d not in dad:
                if random.random() > beta_p:
                    sis.remove(d)
        mom_attitude = random.sample(dad, int(beta * len(dad)))
        for m in mom_attitude:
            if m not in mom:
                if random.random() > beta_p:
                    bro.remove(m)
        children.append(bro)
        children.append(sis)
    else:
        children.append(mom)
        children.append(dad)
    return children


def optimize(imp, pop_size=100, max_generation=100, prng=None):
    """
    my modified nsga-ii
    :param imp: a class IMP instance
    :param pop_size:
    :param max_generation:
    :param prng: I don't know what it is, please check the inspyred doc

    :return: values of final archive set [[influence], [cost]]
    """

    start_time = time()
    # start optimization
    if prng is None:
        prng = random.Random()
        prng.seed(time())

    ea = inspyred.ec.emo.NSGA2(prng)

    ea.variator = [
        my_cross,
        inspyred.ec.variators.random_reset_mutation
    ]
    ea.terminator = inspyred.ec.terminators.generation_termination
    ea.observer = my_observer
    ea.bounder = inspyred.ec.DiscreteBounder(list(range(1, imp.V + 1)))

    ea.evolve(
        evaluator=my_evaluator,
        generator=generator_function,
        pop_size=pop_size,
        maximize=False,
        max_generations=max_generation,
        num_users=imp.V,
        RRS=imp.RRS,
        k=imp.k
    )
    utils.process_end(str(time() - start_time) + 's\n')

    _x = []
    _y = []
    for agent in ea.archive:
        _x.append(imp.IC(agent.candidate, p=imp.p))
        _y.append(agent.fitness[1])

    return [_x, _y]


# test
if __name__ == '__main__':
    pass
