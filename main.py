import numpy

import pandas as pd
from bitmap import bitmap

import greedy
import nsgaii
import utils
from IMP import IMP


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
        result_TIM = greedy.TIM(imp)
        result_EC = nsgaii.optimize(imp, pop_size=50, max_generation=50)

        # visualization
        # utils.show_result([result_EC],
        #                   ["NSGA-II"], name=dataset_name, result_path=imp.result_path)
        utils.show_result([result_TIM, result_EC],
                          ["TIM", "NSGA-II"], name=dataset_name, result_path=imp.result_path)


def test_new_gen():

    # check each dataset

    # create IM problem model
    dataset_name = 'soc-dolphins'
    _k = 10
    imp = IMP(dataset_name,
              k=_k,
              weighted=False,
              directed=False,
              p=0.05
              )
    imp.load_RRS()
    # utils.visual_RRS(imp.G)

    # run algorithms
    result_EC = nsgaii.optimize(imp, pop_size=100, max_generation=10)
    result_CELF = greedy.CELF(imp)
    result_TIM = greedy.TIM(imp)

    # visualization
    # utils.show_result([result_EC],
    #                   ["NSGA-II"], name=dataset_name, result_path=imp.result_path)
    utils.show_result([result_CELF, result_TIM, result_EC],
                      ["CELF", "TIM", "NSGA-II"], name=dataset_name, result_path=imp.result_path)


if __name__ == '__main__':
    test_basic()
    # TODO timer
    # TODO show history population
    # test_new_gen()
