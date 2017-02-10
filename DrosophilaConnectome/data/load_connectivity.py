import csv
import networkx as nx
import os
from os import path

import ipdb

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

def load_flycircuit_graph():
    """ Convert the neuron and synapse files into populations, projections and neurons """

    neuron_fieldnames = ['neuron_name', 'innv_neuropil', 'mem_model', 'resting_pot', 'reset_pot', 'threshold_pot', 'rfact_period', 'Cm', 'tm']

    neuron_property_list = ['resting_pot', 'reset_pot', 'threshold_pot', 'rfact_period', 'Cm', 'tm']

    synapse_fieldnames = ['pre-neuron', 'post-neuron', 'neuropil', 'weight', 'type', 'reversal_pot', 'txt', 'tst', 'pv', 'tD', 'tF', 'tLP', 'PosISI', 'NegISI']

    __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    # Process the text files 
    neuron_reader  = csv.DictReader(open(os.path.join(__location__, 'flycircuit1.2/CUNeuParsV1-2.txt')), fieldnames=neuron_fieldnames,delimiter=' ')
    synapse_reader = csv.DictReader(open(os.path.join(__location__, 'flycircuit1.2/CUSynParsV1-2.txt')), fieldnames=synapse_fieldnames,delimiter=' ')

    neuron_network = nx.DiGraph()
    lpu_network = nx.DiGraph()

    for row in neuron_reader:
        lpu = row['innv_neuropil']
        name = row['neuron_name']
        
        if lpu not in list(lpu_network.nodes()): 
            lpu_network.add_node(lpu)

        neuron_network.add_node(name,attr_dict=row)

    lpu_edges = {}

    for row in synapse_reader:
        pre_neuron = row['pre-neuron']
        post_neuron = row['post-neuron']
        pre_lpu = neuron_network.node[pre_neuron]["attr_dict"]['innv_neuropil']
        post_lpu = neuron_network.node[post_neuron]["attr_dict"]['innv_neuropil']

        neuron_network.add_edge(pre_neuron,post_neuron,attr_dict=row)
        
        if pre_lpu not in lpu_edges:
            lpu_edges[pre_lpu] = {post_lpu : 1}
        else:
            if post_lpu not in lpu_edges[pre_lpu]:
                lpu_edges[pre_lpu][post_lpu] = 1
            else:
                lpu_edges[pre_lpu][post_lpu] += 1

    for pre_lpu in lpu_edges:
        for post_lpu in lpu_edges[pre_lpu]:
            lpu_network.add_edge(pre_lpu, post_lpu, strength=lpu_edges[pre_lpu][post_lpu],distance=1/lpu_edges[pre_lpu][post_lpu])
            

    return neuron_network,lpu_network



