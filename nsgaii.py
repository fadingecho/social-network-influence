import copy
import itertools
import random
from time import time

import inspyred
from inspyred.ec import emo
from inspyred.ec.variators import crossover

import utils


def my_generator(random, args):
    num_users = args.get('num_users')
    return [random.random() for _ in range(num_users)]


def get_cost(seeds):
    rec_cost = 0
    for s in seeds:
        if s > 0.5:
            rec_cost = rec_cost + 1
    return rec_cost


def get_influence(RRS, seeds, args):
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
                if args is not None:
                    si = args.get('special_individual')
                    si[node - 1] = si[node - 1] + 1
                count = count + 1
                break
    return count / len(RRS) * len(seeds)
    # bitmap_seeds = (1 << len(seeds)) - 1
    #
    # count = 0
    # for seed_set in RRS:
    #     for node in seed_set:
    #         if seeds[node - 1] > 0.5 and (bitmap_seeds >> (node - 1)) & 1:
    #             count = count + 1
    #             bitmap_seeds = bitmap_seeds & ~(1 << (node - 1))
    #             break
    # return count / len(RRS) * len(seeds)


def my_evaluator(candidates, args):
    fitness = []
    RRS = args.get('RRS')
    for cs in candidates:
        rec_cost = get_cost(cs)
        influence_spread = get_influence(RRS, cs, args)

        fitness.append(emo.Pareto([-influence_spread, rec_cost]))
    return fitness


def my_bound(candidate, args):
    for _i, c in enumerate(candidate):
        candidate[_i] = max(min(c, 1), 0)
    return candidate


my_bound.lower_bound = itertools.repeat(0)
my_bound.upper_bound = itertools.repeat(1)


def my_observer(population, num_generations, num_evaluations, args):
    max_gen = args.get('max_generations')
    utils.show_process_bar("gen :", num_generations, max_gen)
    if args.get('max_generations') == num_generations:
        utils.process_end("")


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


def optimize(RRS, _user_num, pop_size=100, max_generation=100, prng=None):
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

    special_individual = [0] * _user_num
    ea.evolve(
        evaluator=my_evaluator,
        generator=my_generator,
        bounder=my_bound,
        pop_size=pop_size,
        maximize=False,
        max_generations=max_generation,
        num_users=_user_num,
        RRS=RRS,
        special_individual=special_individual)
    print(str(time() - start_time) + 's\n')

    _x = []
    _y = []
    for agent in ea.archive:
        _x.append(-agent.fitness[0])
        _y.append(agent.fitness[1])

    return [_x, _y], special_individual
