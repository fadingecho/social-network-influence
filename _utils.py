import os
import datetime
import random

from bitmap import BitMap
from matplotlib import pyplot as plt

color_plt = ['b', 'r', 'g', 'y', 'c', 'm']


def show_process_bar(title_str, current_num, total_num):
    print("\r" + title_str + " {:3}/{:3}".format(current_num, total_num), end="")


def process_end(content_str=""):
    print(" " + content_str)


def show_result(result_dict, dataset_name, result_path, title, display=False, xlabel='Influence spread',
                ylabel='Recruitment costs'):
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


def encode_node_ID(filename):
    # create a new file named filename + "-encoded"
    f = '.txt'
    sep = "\t"

    data_file = open(filename + f, 'r')
    id_cnt = 0

    node_dic = dict()
    lines = data_file.readlines()
    new_lines = []
    V_E_line = lines[2].split(" ")
    V, E = V_E_line[2], V_E_line[4]
    new_lines.append("{0}\t{1}".format(V, E))

    for i in range(5, len(lines)):
        n1, n2 = lines[i].strip().split(sep=sep)

        if n1 in node_dic:
            id1 = node_dic[n1]
        else:
            id_cnt += 1
            id1 = id_cnt
            node_dic[n1] = id1

        if n2 in node_dic:
            id2 = node_dic[n2]
        else:
            id_cnt += 1
            id2 = id_cnt
            node_dic[n2] = id2

        new_lines.append("{0}\t{1}\n".format(id1, id2))

    new_file = open(filename + "-encoded" + f, 'w+')
    new_file.writelines(new_lines)

    new_file.close()
    data_file.close()


# draw fig1 for my paper
def fig2():
    """
    raw G       |   sampled G'
    --------------------------
    g(G',v1)    |   g(G',v2)
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    G00 = nx.DiGraph()
    G00.add_edges_from([
        (1, 2), (1, 3), (1, 4), (1, 6),
        (2, 3),
        (3, 5), (3, 6),
        (4, 6),
    ])
    # pos = nx.spectral_layout(G00)
    pos = {
        1: (0, 0), 2: (0.5, 0),
        3: (0, 3), 4: (0.5, 3),
        5: (0, 6), 6: (0.5, 6),
           }

    G10 = nx.DiGraph()
    G10.add_edges_from([(1, 3), (2, 3), (3, 5), (3, 6), (4, 6)])

    G01 = nx.DiGraph()
    G01.add_edges_from([(1, 3), (2, 3), (3, 5), (3, 6), (4, 6)])

    G11 = nx.DiGraph()
    G11.add_edges_from([(1, 3), (2, 3), (3, 5), (3, 6), (4, 6)])

    # Create a 2x2 subplot
    fig, all_axes = plt.subplots(2, 2)
    ax = all_axes.flat

    options = {
        "with_labels": True,
        "font_size": 13,
        "node_size": 500,
        "node_color": "silver",
        "edgecolors": "black",
        "linewidths": 1,
        "width": 1,
    }

    chosen_node_options = {
        "node_size": 500,
        "node_color": "tomato",
        "linewidths": 1,
    }

    other_node_options = {
        "node_size": 500,
        "node_color": "grey",
        "linewidths": 1,
    }

    nx.draw(G00, pos, ax=ax[0], **options)
    nx.draw(G10, pos, ax=ax[1], **options)
    nx.draw(G01, pos, ax=ax[2], **options)
    nx.draw_networkx_nodes(G01, pos, nodelist=[3], ax=ax[2], **chosen_node_options)
    nx.draw_networkx_nodes(G01, pos, nodelist=[1, 2], ax=ax[2], **other_node_options)
    nx.draw(G11, pos, ax=ax[3], **options)
    nx.draw_networkx_nodes(G11, pos, nodelist=[5], ax=ax[3], **chosen_node_options)
    nx.draw_networkx_nodes(G01, pos, nodelist=[1, 2, 3], ax=ax[3], **other_node_options)

    # xlabels
    titles = ["G", "g", "R(g, 3)={1, 2, 3}", "R(g, 5)={1, 2, 3, 5}"]
    # Set margins for the axes so that nodes aren't clipped
    for i in range(len(ax)):
        ax[i].margins(0.3)
        ax[i].set_title(titles[i])
    # ax[0].set_facecolor('lightgoldenrodyellow')

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    # encode_node_ID("./datasets/soc-LiveJournal1")
    fig2()

    # from operator import itemgetter
    #
    # import matplotlib.pyplot as plt
    # import networkx as nx
    #
    # # Create a BA model graph - use seed for reproducibility
    # n = 1000
    # m = 2
    # seed = 20532
    # G = nx.barabasi_albert_graph(n, m, seed=seed)
    #
    # # find node with largest degree
    # node_and_degree = G.degree()
    # (largest_hub, degree) = sorted(node_and_degree, key=itemgetter(1))[-1]
    #
    # # Create ego graph of main hub
    # hub_ego = nx.ego_graph(G, largest_hub)
    #
    # # Draw graph
    # pos = nx.spring_layout(hub_ego, seed=seed)  # Seed layout for reproducibility
    # nx.draw(hub_ego, pos, node_color="b", node_size=50, with_labels=False)
    #
    # options = {"node_size": 300, "node_color": "r"}
    # nx.draw_networkx_nodes(hub_ego, pos, nodelist=[largest_hub], **options)
    # plt.show()