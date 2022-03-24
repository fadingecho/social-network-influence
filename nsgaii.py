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
from spreading_models import IC


def imp_local_search2(population, args):
    # local search
    num_users = args.get('num_users')
    _ec = args.get('_ec')

    seed_mask = bitmap.BitMap(num_users + 1)
    for i in range(1, num_users + 1):
        seed_mask.set(i)

    counter = [0] * (num_users + 1)
    bk = args.get('bk')
    RRS = args.get('RRS')
    ls_skeleton = []
    while not seed_mask.none() and len(ls_skeleton) < int(len(population) * 0.2):
        # we should get a rr set randomly
        r = RRS[random.randint(0, len(RRS) - 1)]
        for node in r:
            if seed_mask.test(node):
                counter[node] = counter[node] + 1
                if counter[node] == bk:
                    ls_skeleton.append(node)
                    seed_mask.reset(node)

    _ec.logger.debug("search result : {0}".format(ls_skeleton))

    ls_offspring_cs = []
    for _ in range(0, int(len(population) * 0.3)):
        cs = random.choice(population).candidate
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
        migrate  : remove 
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


def imp_local_search(population, args):
    # local search
    num_users = args.get('num_users')
    _ec = args.get('_ec')

    seed_mask = bitmap.BitMap(num_users + 1)
    for i in range(1, num_users + 1):
        seed_mask.set(i)

    counter = [0] * (num_users + 1)
    bk = args.get('bk')
    RRS = args.get('RRS')
    ls_skeleton = []
    while not seed_mask.none() and len(ls_skeleton) < int(len(population) * 0.2):
        # we should get a rr set randomly
        r = RRS[random.randint(0, len(RRS) - 1)]
        for node in r:
            if seed_mask.test(node):
                counter[node] = counter[node] + 1
                if counter[node] == bk:
                    ls_skeleton.append(node)
                    seed_mask.reset(node)

    _ec.logger.debug("search result : {0}".format(ls_skeleton))

    ls_offspring_cs = []
    for _ in range(0, int(len(population) * 0.3)):
        cs = random.choice(population).candidate
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
        migrate  : remove 
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

    if args.get("local_search") and num_generations % 5 == 0:
        imp_local_search(population, args)


@crossover
def imp_cross(random, mom, dad, args):
    values = args.get('values')
    num_users = args.get('num_users')

    children = []
    bro = copy.deepcopy(dad)
    sis = copy.deepcopy(mom)

    for i in random.sample(values, int(num_users / 2)):
        if dad.test(i):
            sis.set(i)
        else:
            sis.reset(i)

        if mom.test(i):
            bro.set(i)
        else:
            bro.reset(i)

    children.append(bro)
    children.append(sis)

    return children


@mutator
def imp_mutate(random, candidate, args):
    if random.random() > 0.5:
        return candidate

    num_users = args.get('num_users')
    values = args.get('values')
    for i in random.sample(values, int(num_users * 0.6)):
        candidate.flip(i)
    return candidate


def optimize(imp, pop_size=100, max_generations=100, ls_flag=False, initial_pop=None, prng=None, bk=100):
    """
    my modified nsga-ii
    :param bk:
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
        bk=bk
    )
    _utils.process_end(str(time() - start_time) + 's')

    return ea.archive


# test
if __name__ == '__main__':
    pass
