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


def imp_local_search2(population, args):
    # local search
    _ec = args.get('_ec')

    # chose candidate randomly
    individual_sample = copy.deepcopy(random.sample(population, 50))
    candidate_sample_deep = [i.candidate for i in individual_sample]

    # try to use a small RRS to improve these candidates
    # since the scale of RRS is small, it will converge quickly
    RRS_sample = random.sample(args.get('RRS'), 100)

    # do node selection operation
    hash_table = dict()
    for r in RRS_sample:
        for node in r:
            if node not in hash_table:
                hash_table[node] = []

            hash_table[node].append(r)

    while len(hash_table) > 0:
        # sort the dic by the length of RRS set, which related to the influence
        sorted_table = sorted(hash_table.items(), key=lambda d: len(d[1]), reverse=True)
        max_item = sorted_table[0]
        overlapping_nodes = []

        # record the nodes shares same rr sets with max_item[0]
        # remove the record of overlapped RRS in each item
        trash_can = []
        for r in max_item[1]:
            for node, RRS in hash_table.items():
                if r in RRS:
                    RRS.remove(r)
                    overlapping_nodes.append(node)
                    if len(RRS) == 0:
                        trash_can.append(node)

        # refine the candidate by combine the nodes into max_item[0]
        for c in candidate_sample_deep:
            for ol in overlapping_nodes:
                c.reset(ol)
            c.set(max_item[0])

        for trash in trash_can:
            hash_table.pop(trash)

    maximize = args.get('maximize')
    logger = _ec.logger

    """ copied and modified these below from ec.py : line 478 """
    """ steps : 
        evaluate : local search candidates -----> individuals : (candidates, fitness)
        replace  : put offspring into population, nsga-ii use nd sort and crowding distance to maintain population
        migrate  : remove 
        archive
    """
    # Evaluate offspring.
    logger.debug(
        'evaluation using {0} at generation {1} and evaluation {2}'.format(imp_evaluator.__name__, _ec.num_generations,
                                                                           _ec.num_evaluations))
    ls_offspring_fit = imp_evaluator(candidates=candidate_sample_deep, args=args)
    ls_offspring = []
    for cs, fit in zip(candidate_sample_deep, ls_offspring_fit):
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


def imp_evaluator(candidates, args):
    fitness = []
    RRS = args.get('RRS')
    for cs in candidates:
        # evaluate cost
        rec_cost = cs.count()

        # evaluate influence
        count = 0
        # count the frequency of intersection between the seed set and every rr set
        for R in RRS:
            for r in R:
                if cs.test(r):
                    count = count + 1
                    break
        influence_spread = count / len(RRS) * args.get('num_users')

        fitness.append(emo.Pareto([-influence_spread, rec_cost]))
    return fitness


def imp_observer(population, num_generations, num_evaluations, args):
    max_gen = args.get('max_generations')
    _utils.show_process_bar("nsga-ii evolving :", num_generations, max_gen)

    if num_generations == 0:
        return
    if args.get("local_search") and num_generations % 5 == 0:
        imp_local_search2(population, args)


@crossover
def imp_cross(random, mom, dad, args):
    values = args.get('values')
    num_users = args.get('num_users')

    children = []
    bro = copy.deepcopy(dad)
    sis = copy.deepcopy(mom)

    if random.random() < 1:
        for i in random.sample(values, int(num_users / 2)):
            if dad.test(i):
                sis.set(i)
            else:
                sis.reset(i)

            if mom.test(i):
                bro.set(i)
            else:
                bro.reset(i)

    else:
        # heuristic
        pass

    children.append(bro)
    children.append(sis)
    return children


@mutator
def imp_mutate(random, candidate, args):
    if random.random() < 1:
        return candidate

    num_users = args.get('num_users')
    values = args.get('values')
    for i in random.sample(values, int(num_users * 0.6)):
        candidate.flip(i)
    return candidate


def optimize(imp, pop_size, max_generations, RRS, ls_flag=False, initial_pop=None, prng=None):
    """
    my modified nsga-ii
    :param RRS:
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

    # simply removing or adding nodes will always come with worse combination
    ea.variator = [
        imp_cross,
        # imp_mutate
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
        RRS=RRS,
        k=imp.k,
        local_search=ls_flag,
    )
    _utils.process_end(str(time() - start_time) + 's')

    return ea.archive


# test
if __name__ == '__main__':
    pass
