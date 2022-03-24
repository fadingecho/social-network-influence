import os
import datetime
from bitmap import BitMap
import networkx as nx
from matplotlib import pyplot as plt

color_plt = ['b', 'r', 'g', 'y', 'c', 'm']


def show_process_bar(title_str, current_num, total_num):
    print("\r" + title_str + " {:3}/{:3}".format(current_num, total_num), end="")


def process_end(content_str=""):
    print(" " + content_str)


def show_result(result_dict, dataset_name, result_path, title, display=False, xlabel='Influence spread', ylabel='Recruitment costs'):

    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for key, value in result_dict.items():
        # label : [[[Xs], [Ys]], 'color']
        ax.scatter(value[0][0],
                   value[0][1],
                   s=5,
                   c=value[1],
                   alpha=0.8,
                   label=key)
    ax.legend()
    ax.grid(True)
    plt.title(title)

    # save result, file name identified by a none-sense prefix
    dataset_name = result_path + dataset_name + '/result' + str(datetime.datetime.now().minute) + str(
        datetime.datetime.now().hour) + '.pdf'
    try:
        plt.savefig(dataset_name, format='pdf')
    except FileNotFoundError:
        os.makedirs(result_path + dataset_name)
        plt.savefig(dataset_name, format='pdf')

    if display:
        plt.show()


def encode_set_to_bitmap(S, length):
    bm = BitMap(length + 1)

    # reset bm
    for i in range(1, length + 1):
        if i in set(S):
            bm.set(i)
        else:
            bm.reset(i)

    return bm


def set_ec_logger():
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


def visual_RRS(G, RRS=None):
    # visualize RRS generetated from G
    #
    d_g = nx.DiGraph()

    # step 1 : create directed graph in networkx
    d_g.add_edges_from(G.values.tolist())

    pos = nx.spring_layout(d_g, k=0.5, seed=7355608)
    nx.draw_networkx_nodes(d_g, pos, node_size=100)
    nx.draw_networkx_edges(d_g, pos, alpha=0.4)

    # step 2 : show the results of RRS,
    # nx.draw(d_g, with_labels=True, edge_color='b', node_color='g', node_size=300, font_color='w', font_size=10)
    plt.show()
