import csv
import networkx as nx
import os
from os import path


def load_2015_graph():

    first = True

    network = nx.Graph()
    node_list = []

    __location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

    with open(os.path.join(__location__, '2015paper/connectivity.csv')) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if first:
                for node in row:
                    network.add_node(node)
                    node_list.append(node)
                first = False
            else:
                for i,strength in enumerate(row):
                    if i != 0 and float(strength) != 0.0:
                        network.add_edge(row[0],node_list[i-1],strength=float(strength),distance=1/float(strength))

    return network




