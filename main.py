import pandas as pd
import greedy
import nsgaii
import utils
from IMP import IMP
from My_IMP import My_IMP


def test_basic():
    config_file = pd.read_csv(IMP.datasets_path + "config.csv", delimiter=',', index_col=False, skipinitialspace=True)
    print("read config file")
    print(config_file[['name', 'activate']])
    print("==========")

    for idx in config_file.index:
        # check each dataset
        if not config_file['activate'][idx]:
            continue
        if config_file['weighted'][idx]:
            print("can not process weighted network")
            continue

        # create IM problem model
        dataset_name = str(config_file['name'][idx])

        imp = IMP(dataset_name,
                  k=0.9,
                  weighted=config_file['weighted'][idx],
                  directed=config_file['directed'][idx]
                  )
        imp.load_RRS()
        # utils.visual_RRS(imp.G)

        # run algorithms
        # result_TIM = greedy.TIM(imp, p=0.5)
        result_EC = nsgaii.optimize(imp.RRS, imp.V, pop_size=100, max_generation=50)

        # visualization
        utils.show_result([result_EC],
                          ["NSGA-II"], name=dataset_name, result_path=imp.result_path)
        # utils.show_result([result_TIM, result_EC],
        #                   ["TIM", "NSGA-II"], name=dataset_name, result_path=imp.result_path)


def test_my_thought():
    config_file = pd.read_csv(IMP.datasets_path + "config.csv", delimiter=',', index_col=False, skipinitialspace=True)
    print("read config file")
    print(config_file[['name', 'activate']])
    print("==========")

    for idx in config_file.index:
        # check each dataset
        if not config_file['activate'][idx]:
            continue
        if config_file['weighted'][idx]:
            print("can not process weighted network")
            continue

        # create IM problem model
        dataset_name = str(config_file['name'][idx])

        # imp_approx = My_IMP(dataset_name,
        #                     0.9,
        #                     config_file['weighted'][idx],
        #                     config_file['directed'][idx],
        #                     )
        # imp_approx.load_RRS()

        imp = IMP(dataset_name,

                  k=0.9,
                  weighted=config_file['weighted'][idx],
                  directed=config_file['directed'][idx]
                  )
        imp.load_RRS()
        # utils.visual_RRS(imp.G)

        # run algorithms
        # result_TIM = greedy.TIM(imp, p=0.5)
        result_EC = nsgaii.optimize(imp.RRS, imp.V, pop_size=100, max_generation=50)
        si = result_EC[1]
        si_c = 0
        max_si = max(si)
        for _i in range(len(si)):
            si[_i] = si[_i] / max_si
            if si[_i] > 0.5:
                si_c = si_c + 1
        # result_EC_App = nsgaii.optimize(imp_approx.RRS, imp_approx.V, pop_size=100, max_generation=50)[0]

        # visualization
        utils.show_result([result_EC[0], [[nsgaii.get_influence(imp.RRS, si, None)], [si_c]]],
                          ["NSGA-II", "Approximation"], name=dataset_name, result_path=imp.result_path)


if __name__ == '__main__':
    # test_basic()
    # TODO timer
    test_my_thought()
