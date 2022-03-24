import random

from bitmap import BitMap

import baselines
import nsgaii
import _utils
import solution_evaluation
from IM_problem import IMP


def test_vs_baseline():
    dataset_name = 'soc-wiki-Vote'
    # dataset_name = 'socfb-American75'
    _k = 50
    pop_size = 100
    max_generation = 100

    imp = IMP(dataset_name,
              k=_k,
              weighted=False,
              directed=True,
              mc=200
              )
    theta = baselines.TIM_get_theta(imp)
    imp.load_RRS(theta)

    # prepare seeds
    seeds_list = baselines.TIM(imp)
    seeds_bitmap = []
    for i in range(_k):
        bm = BitMap(imp.V + 1)
        for j in range(1, imp.V + 1):
            if j in set(seeds_list[i]):
                bm.set(j)
            else:
                bm.reset(j)
        seeds_bitmap.append(bm)
    for _ in range(pop_size - _k):
        seed = nsgaii.imp_generator(random, {'num_users': imp.V})
        seeds_bitmap.append(seed)

    moea_archive_TIM_LS = nsgaii.optimize(imp, pop_size=pop_size, max_generations=max_generation, initial_pop=seeds_bitmap, ls_flag=True)
    moea_archive_TIM = nsgaii.optimize(imp, pop_size=pop_size, max_generations=max_generation, initial_pop=seeds_bitmap, ls_flag=False)
    moea_archive_LS = nsgaii.optimize(imp, pop_size=pop_size, max_generations=max_generation, ls_flag=True)
    moea_archive = nsgaii.optimize(imp, pop_size=pop_size, max_generations=max_generation, ls_flag=False)
    random_solutions = baselines.generate_random_solution(100, imp)

    imp.k = imp.V
    trace = baselines.TIM(imp)
    #
    result_TIM = solution_evaluation.RRS_estimate_list(trace, imp)
    result_EC_TIM_LS = solution_evaluation.RRS_estimate_individuals(moea_archive_TIM_LS)
    result_EC_LS = solution_evaluation.RRS_estimate_individuals(moea_archive_LS)
    result_EC_TIM = solution_evaluation.RRS_estimate_individuals(moea_archive_TIM)
    result_EC = solution_evaluation.RRS_estimate_individuals(moea_archive)
    result_random = solution_evaluation.RRS_estimate_list(random_solutions, imp)

    title = dataset_name + " gen : " + str(max_generation) + " pop : " + str(pop_size) + " p : " + str(imp.p) + " RRS estimated"

    # visualization
    _utils.show_result(
        {
            "random": [result_random, 'g'],
            "TIM": [result_TIM, 'b'],
            "NSGA_II_TIM_LS": [result_EC_TIM_LS, 'r'],
            "NSGA_II_TIM": [result_EC_TIM, 'c'],
            "NSGA_II_LS": [result_EC_LS, 'y'],
            "NSGA_II": [result_EC, 'm']
        },
        dataset_name, imp.result_path, title,
        display=True
    )


def test_IMM():
    dataset_name = 'soc-wiki-Vote'
    _k = 10

    imp = IMP(dataset_name,
              k=_k,
              weighted=False,
              directed=True,
              mc=200
              )
    theta = baselines.TIM_get_theta(imp)
    imp.load_RRS(theta)

    IMM_solutions = baselines.IMM(imp)

    #
    result_IMM = solution_evaluation.RRS_estimate_list(IMM_solutions, imp)

    title = dataset_name + " p : " + str(imp.p) + " RRS estimated"
    # visualization
    _utils.show_result(
        {
            "IMM": [result_IMM, 'b'],
        },
        dataset_name, imp.result_path, title,
        display=True
    )


# TODO show history population
if __name__ == '__main__':
    _utils.set_ec_logger()
    # test_vs_baseline()
    test_IMM()