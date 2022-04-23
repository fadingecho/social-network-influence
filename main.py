import datetime
import baselines
import my_evolution
import nsgaii
import _utils
import performance
import solution_evaluation
from memory_profiler import memory_usage
from IM_problem import IMP


def test_optimizers():
    # dataset_name = 'soc-LiveJournal1-encoded'
    dataset_name = 'CA-HepPh-encoded'
    imp = IMP(dataset_name, mc=100, p=0.1)

    # run algorithms
    # MOEAs
    pop_size_1 = 100
    max_generation_1 = 100
    algo1_archive, algo1_info = performance.run_with_measurement(nsgaii.optimize,
                                                                 (imp, pop_size_1, max_generation_1, [imp.reverse_sample_IC() for _ in range(5000)],),
                                                                 {'ls_flag': True, })
    pop_size_2 = 300
    max_generation_2 = 500
    algo2_pop, algo2_info = performance.run_with_measurement(my_evolution.optimize,
                                                             (imp, pop_size_2, max_generation_2, [imp.reverse_sample_IC() for _ in range(500)],),
                                                             {})
    # IMM
    imm_results = []
    for i in range(1, 10):
        imp.set_k(i * 5)
        imm_results.append(baselines.IMM(imp))

    # process result
    # save to files
    exp_info = dataset_name + " gen : " + str(max_generation_1) + " pop : " + str(pop_size_1) + " p : " + str(imp.p)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    _utils.save_solutions({'algo1': (algo1_archive, str(algo1_info)),
                           'algo2': (algo2_pop, str(algo2_info)),
                           'IMM': (imm_results, ' '),
                           },
                          dataset_name, imp.result_path, exp_info,
                          timestamp)

    # evaluate result
    result_algo1 = solution_evaluation.IC_evaluate_individuals_2obj(algo1_archive, imp)
    result_algo2 = solution_evaluation.IC_evaluate_individuals_3obj(algo2_pop, imp)
    result_IMM = solution_evaluation.IC_evaluate_list(imm_results, imp)

    # visualization
    _utils.visualize_result(
        {
            "algo1": [result_algo1, 'g'],
            "algo2": [result_algo2, 'r'],
            "IMM": [result_IMM, 'b']
        },
        dataset_name, imp.result_path, exp_info, timestamp,
        display=True
    )


# TODO print history population
if __name__ == '__main__':
    _utils.set_ec_logger()
    test_optimizers()
