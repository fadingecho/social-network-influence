import copy
import random

import bitmap

import _utils
import inspyred

from inspyred.ec import emo, Individual
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

    # local search
    num_users = args.get('num_users')
    _ec = args.get('_ec')

    pop_sample = random.sample(population, int(len(population) * 0.1))
    node_mask = bitmap.BitMap(num_users + 1)
    for i in range(1, num_users + 1):
        node_mask.reset(i)
        for s in pop_sample:
            if s.candidate.test(i):
                node_mask.set(i)
                break

    counter = [0] * (num_users + 1)
    bk = 10
    RRS = args.get('RRS')
    RRS_it = iter(RRS)
    ls_skeleton = []
    while not node_mask.none() and len(ls_skeleton) <= 10:
        r = next(RRS_it)
        for node in r:
            if node_mask.test(node):
                counter[node] = counter[node] + 1
                if counter[node] == bk:
                    ls_skeleton.append(node)
                    node_mask.reset(node)

    _ec.logger.debug("search result : {0}".format(ls_skeleton))

    ls_offspring_cs = []
    for _ in range(0, 10):
        cs = imp_generator(random, args)
        for sk in random.sample(ls_skeleton, random.randint(1, len(ls_skeleton))):
            cs.reset(sk)
        ls_offspring_cs.append(cs)
    for _ in range(0, 20):
        cs = imp_generator(random, args)
        for sk in random.sample(ls_skeleton, random.randint(1, len(ls_skeleton))):
            cs.set(sk)
        ls_offspring_cs.append(cs)
    _ec.logger.debug(
        'local search generated {0} candidates'.format(len(ls_offspring_cs)))

    maximize = args.get('maximize')
    logger = _ec.logger

    """ copied and modified these below from ec.py : line 478 """
    """ steps : 
        evaluate : local search candidates -----> individuals : (candidates, fitness)
        replace  : put offspring into population, nsga-ii use nd sort and crowding distance to maintain population
        migrate  : 
        archive
    """
    # Evaluate offspring.
    logger.debug(
        'evaluation using {0} at generation {1} and evaluation {2}'.format(imp_evaluator.__name__, _ec.num_generations,
                                                                           _ec.num_evaluations))
    ls_offspring_fit = imp_evaluator(candidates=ls_offspring_cs, args=args)
    ls_offspring = []
    for cs, fit in zip(ls_offspring_cs, ls_offspring_fit):
        if fit is not None:
            off = Individual(cs, maximize=maximize)
            off.fitness = fit
            ls_offspring.append(off)
        else:
            logger.warning('excluding candidate {0} because fitness received as None'.format(cs))
    _ec.num_evaluations += len(ls_offspring_fit)

    # Replace individuals.
    logger.debug('replacement using {0} at generation {1} and evaluation {2}'.format(_ec.replacer.__name__,
                                                                                     _ec.num_generations,
                                                                                     _ec.num_evaluations))
    _ec.population = _ec.replacer(random=random, population=_ec.population, parents=population,
                                  offspring=ls_offspring, args=args)
    logger.debug('population size is now {0}'.format(len(_ec.population)))

    # Migrate individuals.
    logger.debug(
        'migration using {0} at generation {1} and evaluation {2}'.format(_ec.migrator.__name__, _ec.num_generations,
                                                                          _ec.num_evaluations))
    _ec.population = _ec.migrator(random=random, population=_ec.population, args=args)
    logger.debug('population size is now {0}'.format(len(_ec.population)))

    # Archive individuals.
    logger.debug(
        'archival using {0} at generation {1} and evaluation {2}'.format(_ec.archiver.__name__, _ec.num_generations,
                                                                         _ec.num_evaluations))
    _ec.archive = _ec.archiver(random=random, archive=_ec.archive, population=list(_ec.population),
                               args=args)
    logger.debug('archive size is now {0}'.format(len(_ec.archive)))
    logger.debug('population size is now {0}'.format(len(_ec.population)))


@crossover
def imp_cross(random, mom, dad, args):
    values = args.get('values')
    num_users = args.get('num_users')

    crossover_rate = args.setdefault('crossover_rate', 1.0)
    children = []
    if random.random() < crossover_rate:
        bro = copy.deepcopy(dad)
        sis = copy.deepcopy(mom)
        for i in random.sample(values, random.randint(100, num_users)):
            if dad.test(i):
                sis.set(i)
            else:
                sis.reset(i)
            if mom.test(i):
                bro.set(i)
            else:
                bro.reset(i)

        # cross over
        # dad_attitude = random.sample(values, int(alpha * num_users))
        # for d in dad_attitude:
        #     if dad.test(d) and not mom.test(d):
        #         if random.random() > alpha_p:
        #             sis.set(d)
        #     elif not dad.test(d) and mom.test(d):
        #         if random.random() > alpha_p:
        #             sis.reset(d)
        #
        # mom_attitude = random.sample(values, int(alpha * num_users))
        # for m in mom_attitude:
        #     if mom.test(m) and not dad.test(m):
        #         if random.random() > alpha_p:
        #             bro.set(m)
        #     elif not mom.test(m) and dad.test(m):
        #         if random.random() > alpha_p:
        #             bro.reset(m)

        children.append(bro)
        children.append(sis)
    else:
        children.append(mom)
        children.append(dad)
    return children


@mutator
def imp_mutate(random, candidate, args):
    num_users = args.get('num_users')
    values = args.get('values')
    for i in random.sample(values, random.randint(100, num_users)):
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
    )
    _utils.process_end(str(time() - start_time) + 's')

    return ea.archive


# test
if __name__ == '__main__':
    pass
