import copy
import random

import bitmap

import _utils
import inspyred

from inspyred.ec import emo
from inspyred.ec.generators import diversify
from inspyred.ec.variators import crossover, mutator
from bitmap import BitMap
from time import time


@diversify
def imp_generator(random, args):
    num_users = args.get('num_users')
    bm = BitMap(num_users + 1)

    # reset bm
    for i in range(1, num_users + 1):
        bm.reset(i)

    for i in random.sample(range(1, num_users + 1), random.randint(1, num_users)):
        # flip coins randomly at random times
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
    _utils.show_process_bar("nsga-ii evolving :", num_generations, max_gen)


@crossover
def imp_cross(random, mom, dad, args):
    alpha = 0.3
    alpha_p = 0.2
    values = args.get('values')
    num_users = args.get('num_users')

    crossover_rate = args.setdefault('crossover_rate', 1.0)
    children = []
    if random.random() < crossover_rate:
        bro = copy.deepcopy(dad)
        sis = copy.deepcopy(mom)

        # cross over
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
        if random.random() < 0.3:
            candidate.flip(i)
    return candidate


def IC_evaluate_archive(BM, imp):
    # evaluate archive
    influence = []
    cost = []
    _utils.show_process_bar("IC evaluating ", 0, len(BM))
    for _i in range(len(BM)):
        _utils.show_process_bar("IC evaluating ", _i, len(BM))
        # construct solution
        solution = []
        candidate = BM[_i].candidate
        for i in range(1, candidate.size()):
            if candidate.test(i):
                solution.append(i)
        influence.append(imp.IC(solution))
        cost.append(BM[_i].fitness[1])
    _utils.process_end("")
    return [influence, cost]


def optimize(imp, pop_size=100, max_generations=100, ls_flag=False, initial_pop=None, prng=None):
    """
    my modified nsga-ii
    :param initial_pop: if initial population is given, it will be used
    :param ls_flag: local search option
    :param imp: a class IMP instance
    :param pop_size:
    :param max_generations:
    :param prng: pseudo-random number generator, save it for restarting the same run

    :return: values of final archive set [[influence], [cost]]
    """

    start_time = time()

    # parameter setting
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

    # optimize start from here
    ea.evolve(
        evaluator=imp_evaluator,
        generator=imp_generator,
        seeds=initial_pop,
        pop_size=pop_size,
        max_generations=max_generations,
        maximize=False,
        values=range(1, imp.V + 1),  # for discrete bounder

        num_users=imp.V,
        RRS=imp.RRS,
        k=imp.k,
        local_search=ls_flag,
        imp=imp
    )
    _utils.process_end(str(time() - start_time) + 's')

    return ea.archive


# test
if __name__ == '__main__':
    pass
