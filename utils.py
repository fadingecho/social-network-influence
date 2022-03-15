import numpy as np
import os
import datetime
from matplotlib import pyplot as plt, pylab
import networkx as nx


def show_process_bar(title_str, current_num, total_num):
    print("\r" + title_str + " {:3}/{:3}".format(current_num, total_num), end="")


def process_end(content_str=""):
    print(" " + content_str)


def show_result(results, labels, name, result_path, display=False):
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

    # save result, file name identified by a none-sense prefix
    file_name = result_path + name + '/result' + str(datetime.datetime.now().minute) + str(
        datetime.datetime.now().hour) + '.pdf'
    try:
        plt.savefig(file_name, format='pdf')
    except FileNotFoundError:
        os.makedirs(result_path + name)
        plt.savefig(file_name, format='pdf')

    if display:
        plt.show()


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
