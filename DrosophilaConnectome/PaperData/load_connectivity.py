import csv
import networkx as nx

def load_2015_graph():

    first = True

    network = nx.Graph()
    node_list = []
    with open('connectivity.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if first:
                for node in row:
                    network.add_node(node)
                    node_list.append(node)
                first = False
            else:
                for i,strength in enumerate(row):
                    if i != 0:
                        network.add_edge(row[0],node_list[i-1],strength=float(strength),distance=1/float(strength))



