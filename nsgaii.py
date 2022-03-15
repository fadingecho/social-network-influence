import copy
import random
from time import time

import bitmap
import inspyred
from inspyred.ec import emo
from inspyred.ec.generators import diversify
from inspyred.ec.variators import crossover, mutator
from bitmap import BitMap

import utils


@diversify
def imp_generator(random, args):
    num_users = args.get('num_users')
    bm = BitMap(num_users + 1)
    for i in range(1, num_users + 1):
        bm.reset(i)

    for i in random.sample(range(1, num_users + 1), random.randint(1, num_users)):
        bm.flip(i)
    return bm


def imp_cost(seeds):
    return seeds.count()


def imp_influence(RRS, seeds, args):
    """
        Inputs: RRS: random RR set
                seeds: a seed set of nodes, represented by a node set
                num_nodes: number of the nodes in the network
        Return: a value represents the estimation of influence of the seeds
    """

    count = 0
    # count the frequency of intersection between the seed set and every rr set
    for R in RRS:
        for r in R:
            if seeds.test(r):
                count = count + 1
                break
    return count / len(RRS) * args.get('num_users')


def imp_evaluator(candidates, args):
    fitness = []
    RRS = args.get('RRS')
    for cs in candidates:
        rec_cost = imp_cost(cs)
        influence_spread = imp_influence(RRS, cs, args)

        fitness.append(emo.Pareto([-influence_spread, rec_cost]))
    return fitness


def imp_observer(population, num_generations, num_evaluations, args):
    max_gen = args.get('max_generations')
    utils.show_process_bar("nsga-ii evolving :", num_generations, max_gen)


# TODO sample from flat RRS
@crossover
def imp_cross(random, mom, dad, args):
    alpha = 0.3
    alpha_p = 0.5
    values = args.get('values')
    num_users = args.get('num_users')

    crossover_rate = args.setdefault('crossover_rate', 1.0)
    children = []
    if random.random() < crossover_rate:
        bro = copy.copy(dad)
        sis = copy.copy(mom)

        # add node
        dad_attitude = random.sample(values, int(alpha * num_users))
        for d in dad_attitude:
            if dad.test(d) and not mom.test(d):
                if random.random() > alpha_p:
                    sis.set(d)
            elif not dad.test(d) and mom.test(d):
                if random.random() > alpha_p:
                    sis.reset(d)
        mom_attitude = random.sample(values, int(alpha * num_users))
        for m in mom_attitude:
            if mom.test(m) and not dad.test(m):
                if random.random() > alpha_p:
                    bro.set(m)
            elif not mom.test(m) and dad.test(m):
                if random.random() > alpha_p:
                    bro.reset(m)

        children.append(bro)
        children.append(sis)
    else:
        children.append(mom)
        children.append(dad)
    return children


@mutator
def imp_mutate(random, candidate, args):
    num_users = args.get('num_users')
    for i in range(1, num_users + 1):
        if random.random() < 0.1:
            candidate.flip(i)
    return candidate


def optimize(imp, pop_size=100, max_generation=100, ls=False, prng=None):
    """
    my modified nsga-ii
    :param ls: local search option
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
        imp_cross,
        imp_mutate
    ]
    ea.terminator = inspyred.ec.terminators.generation_termination
    ea.observer = imp_observer
    ea.bounder = inspyred.ec.DiscreteBounder(list(range(1, imp.V + 1)))

    ea.evolve(
        evaluator=imp_evaluator,
        generator=imp_generator,
        pop_size=pop_size,
        maximize=False,
        max_generations=max_generation,
        num_users=imp.V,
        RRS=imp.RRS,
        k=imp.k,
        values=range(1, imp.V + 1)
    )
    utils.process_end(str(time() - start_time) + 's')

    # evaluate archive
    influence = []
    cost = []
    utils.show_process_bar("nsga-ii IC ", 0, len(ea.archive))
    for _i in range(len(ea.archive)):
        utils.show_process_bar("nsga-ii IC ", _i, len(ea.archive))
        # construct solution
        solution = []
        candidate = ea.archive[_i].candidate
        for i in range(1, candidate.size()):
            if candidate.test(i):
                solution.append(i)
        influence.append(imp.IC(solution))
        cost.append(ea.archive[_i].fitness[1])
    utils.process_end("")
    return [influence, cost]


# test
if __name__ == '__main__':
    pass
