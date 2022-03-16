import baseline_random
import greedy
import nsgaii
import _utils
from IM_problem import IMP

import pandas as pd


def test_diff_IC_RRS():
    config_file = pd.read_csv(IMP.datasets_path + "config.csv", delimiter=',', index_col=False, skipinitialspace=True)
    print("config file")
    print(config_file[['name', 'activate']])
    print("")

    for idx in config_file.index:
        # check each dataset
        if not config_file['activate'][idx]:
            continue
        if config_file['weighted'][idx]:
            print("can not process weighted network")
            continue

        # create IM problem model
        dataset_name = str(config_file['name'][idx])
        print('-----------' + dataset_name + '-----------')

        imp = IMP(dataset_name,
                  k=50,
                  weighted=config_file['weighted'][idx],
                  directed=config_file['directed'][idx]
                  )
        imp.load_RRS()

        # get seeds returned by tim
        trace, result_TIM = greedy.TIM(imp)
        _cost = []
        _influence = []
        for t in trace:
            count = 0
            # count the frequency of intersection between the seed set and every rr set
            for R in imp.RRS:
                if len(set(R) & set(t)) != 0:
                    count = count + 1
            _influence.append(count / len(imp.RRS) * imp.V)
            _cost.append(len(t))
        result_RRS = [_influence, _cost]

        # visualization
        title = dataset_name + " p : " + str(imp.p)
        _utils.show_result({"IC": [result_TIM, 'r'], "RRS": [result_RRS, 'b']},
                           dataset_name=title, result_path=imp.result_path, display=True)


def test_RRS_evaluation():
    dataset_name = 'soc-wiki-Vote'
    _k = 100
    pop_size = 50
    max_generation = 10

    imp = IMP(dataset_name,
              k=_k,
              weighted=False,
              directed=True,
              )
    imp.load_RRS()

    # run algorithms
    trace = greedy.TIM(imp)
    result_TIM = [[], []]
    for t in trace:
        # evaluate
        count = 0
        for r in imp.RRS:
            for v in t:
                if v in set(r):
                    count = count + 1
                    break
        result_TIM[0].append(count / len(imp.RRS) * imp.V)
        result_TIM[1].append(len(t))

    moea_archive = nsgaii.optimize(imp, pop_size=pop_size, max_generations=max_generation)
    result_EC = [[-a.fitness[0] for a in moea_archive], [a.fitness[1] for a in moea_archive]]

    # visualization
    title = dataset_name + "  gen : " + str(max_generation) + " p : " + str(imp.p) + " evaluated by RRS"
    _utils.show_result(
        {
            "NSGA-II": [result_EC, 'r'],
            "TIM": [result_TIM, 'b'],
        },
        dataset_name, imp.result_path, title,
        display=True
    )


def test_basic():
    config_file = pd.read_csv(IMP.datasets_path + "config.csv", delimiter=',', index_col=False, skipinitialspace=True)
    print("config file")
    print(config_file[['name', 'activate']])
    print("")

    for idx in config_file.index:
        # check each dataset
        if not config_file['activate'][idx]:
            continue
        if config_file['weighted'][idx]:
            print("can not process weighted network")
            continue

        # create IM problem model
        dataset_name = str(config_file['name'][idx])
        print('-----------' + dataset_name + '-----------')

        imp = IMP(dataset_name,
                  weighted=config_file['weighted'][idx],
                  directed=config_file['directed'][idx],
                  k=50,
                  mc=100
                  )
        imp.load_RRS()

        # run algorithms
        pop_size = 100
        max_generation = 100

        result_random = baseline_random.random_solutions(500, imp)
        trace, result_TIM = greedy.TIM(imp)
        result_EC = nsgaii.optimize(imp, pop_size=pop_size, max_generation=max_generation)

        # visualization
        # _utils.show_result({"NSGA-II": result_EC},
        #                    name=dataset_name, result_path=imp.result_path, display=True)
        title = dataset_name + "p : " + str(imp.p) + " gen : " + str(max_generation)
        _utils.show_result({"TIM": [result_TIM, 'r'], "NSGA-II": [result_EC, 'g'], "Random": [result_random, 'b']},
                           dataset_name=dataset_name, result_path=imp.result_path, title=title,
                           display=True)


def test_vs_random():
    # check each dataset
    # create IM problem model
    dataset_name = 'soc-wiki-Vote'
    _k = 100
    pop_size = 50
    max_generation = 50

    imp = IMP(dataset_name,
              k=_k,
              weighted=False,
              directed=True,
              mc=100
              )
    imp.load_RRS()

    # run algorithms
    trace = greedy.TIM(imp)
    # seeds = [_utils.encode_set_to_bitmap(t, imp.V) for t in trace]
    # result_TIM = greedy.IC_evaluate(trace, imp)

    moea_archive = nsgaii.optimize(imp, pop_size=pop_size, max_generations=max_generation)
    result_EC = nsgaii.IC_evaluate_archive(moea_archive, imp)

    # moea_archive_TIM = nsgaii.optimize(imp, pop_size=imp.k, max_generations=max_generation, initial_pop=trace)
    # result_EC_TIM = nsgaii.IC_evaluate_archive(moea_archive_TIM, imp)

    result_random = baseline_random.random_solutions(100, imp)

    # visualization
    title = dataset_name + " gen : " + str(max_generation) + " p : " + str(imp.p)
    _utils.show_result(
        {
            "NSGA-II": [result_EC, 'r'],
            "random": [result_random, 'g'],
            # "TIM": [result_TIM, 'b'],
            # "NSGA_II_TIM": [result_EC_TIM, 'k']
        },
        dataset_name, imp.result_path, title,
        display=True)


def test_TIM_initial():
    dataset_name = 'soc-wiki-Vote'
    _k = 100
    pop_size = 50
    max_generation = 10

    imp = IMP(dataset_name,
              k=_k,
              weighted=False,
              directed=True,
              mc=100
              )
    imp.load_RRS()

    # run algorithms
    trace = greedy.TIM(imp)
    result_TIM = greedy.IC_evaluate(trace, imp)
    # result_EC_ls = nsgaii.optimize(imp, pop_size=pop_size, max_generations=max_generation,
    #                                initial_pop=[_utils.encode_set_to_bitmap(t, imp.V) for t in trace])
    result_EC = nsgaii.optimize(imp, pop_size=pop_size, max_generations=max_generation)
    result_random = baseline_random.random_solutions(200, imp)

    # visualization
    title = dataset_name + " gen : " + str(max_generation) + " p : " + str(imp.p)
    _utils.show_result(
        {
            "NSGA-II": [result_EC, 'r'],
            # "NSGA-II(initiated by TIM)": [result_EC_ls, 'g'],
            "TIM": [result_TIM, 'b'],
            "random": [result_random, 'm']
        },
        dataset_name, imp.result_path, title,
        display=True
    )


if __name__ == '__main__':
    # Use and Consult the Logs
    # Check : https://pythonhosted.org/inspyred/troubleshooting.html#use-and-consult-the-logs
    import logging

    logger = logging.getLogger('inspyred.ec')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler('inspyred.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # test_basic()
    # TODO timer
    # TODO show history population
    # test_ls()
    # test_diff_IC_RRS()
    # test_TIM_initial()
    # test_RRS_evaluation()
    test_vs_random()
