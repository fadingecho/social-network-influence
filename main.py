import datetime
import random

import numpy as np
from bitmap import BitMap

import baselines
import my_evolution
import nsgaii
import _utils
import solution_evaluation
from IM_problem import IMP


def test_optimizers():
    # dataset_name = 'soc-LiveJournal1-encoded'
    dataset_name = 'CA-HepPh-encoded'
    imp = IMP(dataset_name, mc=1, p=0.1, k=0)

    #
    # imm_solutions = baselines.IMM_points(imp, 10)
    # np.savetxt('IMM-0.02.txt', imm_solutions, delimiter=", ", fmt="% s")

    # imp.refresh_rrset()

    pop_size = 1
    max_generation = 1
    # moea_archive_LS = nsgaii.optimize(imp, pop_size=pop_size, max_generations=max_generation, ls_flag=True, RRS=[imp.get_a_unused_rrset() for _ in range(100000)])
    moea_final_pop = my_evolution.optimize(imp, pop_size=pop_size, max_generations=max_generation, RRS=[imp.get_a_unused_rrset() for _ in range(2000)])

    exp_info = dataset_name + " gen : " + str(max_generation) + " pop : " + str(pop_size) + " p : " + str(imp.p)
    str_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    # save results
    _utils.save_results({'NSGA_II_dy': moea_final_pop}, dataset_name, imp.result_path, exp_info, str_time)

    #
    # result_EC = solution_evaluation.RRS_estimate_individuals(moea_archive_LS, imp, test_RRS)
    result_EC_2 = solution_evaluation.IC_evaluate_individuals_2(moea_final_pop, imp)
    # result_IMM = solution_evaluation.IC_evaluate_list(imm_solutions, imp)

    # visualization
    _utils.visualize_result(
        {
            # "NSGA-II-ls": [result_EC, 'g'],
            "NSGA_II_dy": [result_EC_2, 'r'],
            # "IMM": [result_IMM, 'b']
        },
        dataset_name, imp.result_path, exp_info, str_time,
        display=True
    )

    # TODO : save result to txt

    imp.save_new_rrset()


# TODO show history population
if __name__ == '__main__':
    _utils.set_ec_logger()
    test_optimizers()

    # dataset_name = 'soc-LiveJournal1-encoded'
    # # dataset_name = 'soc-douban'
    # pop_size = 100
    # max_generation = 500
    #
    # imp = IMP(dataset_name, mc=200, p=0.1, k=0)
    #
    # for _ in range(2000):
    #     imp.get_a_unused_rrset()
    #     imp.save_new_rrset()



