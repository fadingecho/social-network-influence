import os
import datetime
import pandas as pd
from matplotlib import pyplot as plt, pylab
import greedy
import nsgaii
from IMP import my_IM


def show_result(results, labels, name, result_path):
    color = ['b', 'r', 'g']

    fig, ax = plt.subplots()
    ax.set_xlabel('Influence spread')
    ax.set_ylabel('Recruitment costs')
    for _i in range(len(results)):
        ax.scatter(results[_i][0],
                   results[_i][1],
                   s=3,
                   c=color[_i],
                   alpha=0.8,
                   label=labels[_i])
    ax.legend()
    ax.grid(True)
    plt.title(name)
    # plt.show()

    # save result, file name identified by a none-sense prefix
    file_name = result_path + name + '/result' + str(datetime.datetime.now().minute) + str(datetime.datetime.now().hour) + '.pdf'
    try:
        pylab.savefig(file_name, format='pdf')
    except FileNotFoundError:
        os.makedirs(result_path + name)
        pylab.savefig(file_name, format='pdf')


if __name__ == '__main__':
    config_file = pd.read_csv(my_IM.datasets_path + "config.csv", delimiter=',', index_col=False, skipinitialspace=True)
    print("read config file")
    print(config_file[['name', 'activate']])
    print("==========")

    for idx in config_file.index:
        # check each dataset
        if not config_file['activate'][idx]:
            continue
        if config_file['weighted'][idx]:
            print("can not process weight network")
            continue

        # create IM problem model
        dataset_name = str(config_file['name'][idx])
        imp = my_IM(dataset_name, weighted=config_file['weighted'][idx], directed=config_file['directed'][idx])

        # run algorithms
        result_TIM = greedy.TIM(imp, p=0.5)
        result_EC = nsgaii.optimize(imp.RRS, imp.V, pop_size=100, max_generation=100)

        # visualization
        show_result([result_TIM, result_EC],
                    ["TIM", "NSGA-II"], name=dataset_name, result_path=imp.result_path)
