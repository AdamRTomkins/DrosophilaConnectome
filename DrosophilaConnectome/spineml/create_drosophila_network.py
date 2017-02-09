"""
A script to convert the drosophila connectome into SpineML 
 
# Install libSpineML from source
# https://github.com/AdamRTomkins/libSpineML

"""
from __future__ import division

from libSpineML import smlExperiment as exp
from libSpineML import smlNetwork as net
from libSpineML import smlComponent as com

import csv
import sys

import cStringIO
import graphviz as gv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


import copy

neuron_fieldnames = ['neuron_name', 'innv_neuropil', 'mem_model', 'resting_pot', 'reset_pot', 'threshold_pot', 'rfact_period', 'Cm', 'tm']

neuron_property_list = ['resting_pot', 'reset_pot', 'threshold_pot', 'rfact_period', 'Cm', 'tm']


synapse_fieldnames = ['pre-neuron', 'post-neuron', 'neuropil', 'weight', 'type', 'reversal_pot', 'txt', 'tst', 'pv', 'tD', 'tF', 'tLP', 'PosISI', 'NegISI']


def main():
    """ Process the parameter files and create a SpineML network """

    print "Processing Files..."
    neurons, populations, projections  = process_files('CUNeuParsV1-2.txt','CUSynParsV1-2.txt')

    print "Creating SpineML representation..."
    create_spineml_network(neurons, populations,
        projections,output_filename='model.xml',project_name= 'drosophila')
  
    print "Creating Graph Visualisation..."
    create_graphviz_graph(populations,projections)
  

def process_files(neuron_file,synapse_file):
    """ Convert the neuron and synapse files into populations, projections and neurons """

    # Process the text files 
    neuron_reader = csv.DictReader(open('CUNeuParsV1-2.txt'), fieldnames=neuron_fieldnames,delimiter=' ')
    synapse_reader = csv.DictReader(open('CUSynParsV1-2.txt'), fieldnames=synapse_fieldnames,delimiter=' ')

    neurons = {}
    populations = {}
    projections = {}


    for row in neuron_reader:
        lpu = row['innv_neuropil']
        name = row['neuron_name']
        
        if lpu not in populations: 
            populations[lpu] = [name]

        else:    
            populations[lpu].append(name)

        neurons[name] = row
        neurons[name]['index']= len(populations[lpu])-1


    for row in synapse_reader:
        pre_neuron = row['pre-neuron']
        post_neuron = row['post-neuron']
        
        # get the LPU of the pre neuron
        pre_lpu = neurons[pre_neuron]['innv_neuropil']
        # get the LPU index of the pre neuron
        pre_index = neurons[pre_neuron]['index']
        # get the LPU of the post neuron
        post_lpu = neurons[post_neuron]['innv_neuropil']
        # get the LPU index of the post neuron
        post_index = neurons[post_neuron]['index']

        if pre_lpu not in projections: 
            projections[pre_lpu] = {}

        if post_lpu not in projections[pre_lpu]: 
            projections[pre_lpu][post_lpu] = []

        projections[pre_lpu][post_lpu].append((pre_index,post_index))
        
    return (neurons, populations, projections)

def create_spineml_network(neurons, populations,   
    projections,output_filename='model.xml',project_name= 'drosophila'):
    """ convert projections and populations into a SpineML network """

    # create the network SpineML type
    network = net.SpineMLType()

    # for each population, create a Population type 
    for p in populations:
        population = net.PopulationType()

        # create a neuron type
        neuron = net.NeuronType()
        n = neurons.keys()[0] # The model neuron to use as the template

        # Build this Neuron Sets Property list
        # Currently all fixed value # TODO
        for np in neuron_property_list:
            value = net.FixedValueType(neurons[n][np]) # Currently using a fixed value, should use valuelist
            name = np
            dimension = '?'#Todo Add dimensions to property list 
            neuron_property = net.PropertyType()
            neuron_property.set_name(name)
            neuron_property.set_dimension(dimension)
            neuron_property.set_AbstractValue(value)
            neuron.add_Property(neuron_property)
        neuron.set_name(p)        
        neuron.set_url(neurons[n]['mem_model']+'.xml')
        neuron.set_size(len(populations[p])) 

        # Assign to population
        population.set_Neuron(neuron)

        # create a projection        
        
        for destination in projections[p]:
            projection = net.ProjectionType(destination)
            # Add synapses
            synapse = net.SynapseType()
            connection_list = net.ConnectionListType()
            for connection in projections[p][destination]:
           
                connection = net.ConnectionType(connection[0],connection[1],0) # zero delay
                connection_list.add_Connection(connection)

            synapse.set_AbstractConnection(connection)
            projection.add_Synapse(synapse)

            population.add_Projection(projection)
        
        # add population to the network
        network.add_Population(population)
        
    # Write out network to xml
    io = cStringIO.StringIO()
    network.export(io,0)
    io = io.getvalue()

    # Cleanup Replace Abstract objects with non_abstract
    subs = {
        "AbstractConnection":"ConnectionList",
        "AbstractValue":"FixedValue",
        "Population":"LL:Population",
        "Neuron":"LL:Neuron",
        "Projection":"LL:Projection",
        "Synapse":"LL:Synapse",
        "WeightUpdate":"LL:WeightUpdate",
        '<SpineMLType>':
        '<LL:SpineML xsi:schemaLocation="http://www.shef.ac.uk/SpineMLLowLevelNetworkLayer SpineMLLowLevelNetworkLayer.xsd http://www.shef.ac.uk/SpineMLNetworkLayer SpineMLNetworkLayer.xsd" name="%s">' % project_name,
        '</SpineMLType>':'</LL:SpineML>'
    }

    for k in subs:
        io = io.replace(k,subs[k])

    with open(output_filename, 'w') as f:
        f.write(io)

    return io

def create_graphviz_graph(populations,projections):
    """ convert the projections matrix to a svg graph """

    g1 = gv.Digraph(format='svg')
    for lpu in populations.keys():
        if lpu.lower() == lpu:
            g1.node(lpu)

    for pre in projections.keys():
        if pre.lower() == pre:
            for post in projections[pre]:
                if post.lower() == post:
                    if len(projections[pre][post]) > 100:         
                        g1.edge(pre, post,weight = str(len(projections[pre][post])))


    filename = g1.render(filename='left_hemisphere')
    print filename

def create_networkx_graph(populations,projections,prune=10):
    """ convert the projections matrix to a svg graph """

    network = nx.Graph()
    lpus = populations.keys()
    for lpu in lpus:
        network.add_node(lpu)

    for pre in projections.keys():
        for post in projections[pre]:
            if len(projections[pre][post]) > prune:         
                network.add_edge(pre, post, weight=1.0/len(projections[pre][post]))

    """
    nx.draw_circular(network)
    plt.savefig("graph.png")
    plt.show()
    """
    return network

def basic_measurement_strength(populations,projections,plot=False):
    """ calculate the strength of each LPU.
        Strength is calculated as the total sum of incoming and outgoing 
        connections to an LPU
    """
    measurements = {"strength":{},"reciever":{},"sender":{},'polarity':{},'degree':{},'edge_strength':{},'edge_polarity':{}} 
    for lpu in populations.keys():
        measurements["strength"][lpu] = 0
        measurements["reciever"][lpu] = 0
        measurements["sender"][lpu] = 0
        measurements["polarity"][lpu] = 0
        measurements["degree"][lpu] = 0

    for pre in projections:
        for post in projections[pre]:
            if pre != post:
                measurements['degree'][pre] += 1
                measurements['degree'][post] += 1
                measurements['strength'][pre] += len(projections[pre][post])
                measurements['strength'][post] += len(projections[pre][post])
                measurements['sender'][pre] += len(projections[pre][post])
                measurements['reciever'][post] += len(projections[pre][post])

                #Calculate Edge Strength
                if pre not in measurements['edge_strength']:
                    measurements['edge_strength'][pre] = {}
                measurements['edge_strength'][pre][post] = len(projections[pre][post])

    for lpu in populations.keys():
        measurements["polarity"][lpu] = (measurements["reciever"][lpu] - measurements["sender"][lpu]) / (measurements["reciever"][lpu] + measurements["sender"][lpu])

    for pre in projections:
        for post in projections[pre]:
            try:
                send_str =  measurements['edge_strength'][pre][post]
            except:
                 send_str = 0

            try:
                recieve_str =  measurements['edge_strength'][post][pre]
            except:
                recieve_str = 0
            
            if pre not in measurements['edge_polarity']:
                measurements['edge_polarity'][pre] = {post:0}
            else:
                if post not in measurements['edge_polarity'][pre]:
                    measurements['edge_polarity'][pre][post] = 0
 
            measurements['edge_polarity'][pre][post] = (np.abs(send_str - recieve_str))/(send_str+recieve_str)


    if plot:
        plot_matrix(measurements['edge_polarity'],populations.keys())
        plot_measurement(measurements['strength'],"LPU", "Strength")
        plot_measurement(measurements['polarity'],"LPU", "Polarity")
        plot_measurement(measurements['degree'],"LPU", "Degree")



    return measurements



def network_global_efficiency(network):
    """   """

    lpus = network.nodes()

    m_size = len(lpus)
    sp = nx.shortest_path(network,weight='weight')
    shortest_paths = np.zeros((m_size,m_size))

    for pre in sp.keys():
        for post in sp[pre].keys():
            path = sp[pre][post]
            if len(path) > 1:
                for i in np.arange(len(path)-1):
                    shortest_paths[lpus.index(post),lpus.index(pre)] += (network.edge[path[i]][path[i+1]]['weight'])

    median_path = np.median(shortest_paths)
    e_temp = 0
    for i in np.arange(len(lpus)):
        for j in np.arange(len(lpus)):
            if i != j:
                if shortest_paths[j,i] !=0:
                    e_temp += 1.0/shortest_paths[j,i]

    global_efficiency  = 1.0/(m_size*(m_size-1)) * e_temp
    return (global_efficiency, median_path,shortest_paths)    

def network_vulnerability(network):
    """ calculate the vulnerability of each node """

    # get the global stats
    global_efficiency, median_path, shortest_paths = network_global_efficiency(network)
    lpus = network.nodes()
    measures = {"vulnerability":{},"median_path_length":{}}

    # for each lpu, remove it, and recalculate
    for i, lpu in enumerate(lpus):
        print "remove " + lpu
        tmp_network = copy.deepcopy(network)
        tmp_network.remove_node(lpu)
        print len(tmp_network.nodes()) - len(network.nodes())
        print len(tmp_network.edges()) - len(network.edges())

        tmp_global_efficiency, tmp_median_path,_  = network_global_efficiency(tmp_network)
        
        measures["vulnerability"][lpu]          = global_efficiency - tmp_global_efficiency
        measures["median_path_length"][lpu]     = median_path - tmp_median_path
        
        print str(tmp_global_efficiency - global_efficiency)

    plot_measurement(measures["vulnerability"],xlabel= "LPU",ylabel="Vulnerability")
    plot_measurement(measures["median_path_length"],xlabel= "LPU",ylabel="Median Path Length")

    return (measures, network.nodes())

def contribution(network):
    """ calculate the contribution of the outgoing weights from one lpu """
    l = network.nodes()
    for lpu_pre in l:
        c = 0
        for lpu_post in l:
            try:
                c = c + network.edge[lpu_pre][lpu_post]['weight']
            except:
                pass
        contribution[lpu_pre] =  c
    return contribution

def betweenness_centrality(network,weighted=False):

    betweenness = {}
    nodes = network.nodes()

    shortest_paths = {}
    for pre_lpu in nodes:
        shortest_paths[pre_lpu] = {}
        for post_lpu in nodes:
            if weighted:
                shortest_paths[pre_lpu][post_lpu] = list(nx.all_shortest_paths(network,pre_lpu,post_lpu,weight='weight'))
            else:
                shortest_paths[pre_lpu][post_lpu] = list(nx.all_shortest_paths(network,pre_lpu,post_lpu))

    for n in nodes:
        b = 0
        for pre_lpu in nodes:
            for post_lpu in nodes:
                if pre_lpu != post_lpu and pre_lpu != n and post_lpu !=n:
                    p = 0
                    pi = 0
                    for path in shortest_paths[pre_lpu][post_lpu]:
                        p+=1;
                        pi += n in path

                    ratio = float(pi)/p
                    b = b+ ratio 
        betweenness[n] = b
    return betweenness
    


if __name__ == "__main__":
    #main()
    neurons, populations, projections  = process_files('CUNeuParsV1-2.txt','CUSynParsV1-2.txt')

    #m = basic_measurement_strength(populations,projections,True)
    #graph = create_networkx_graph(populations,projections,prune=10)
    #shortest_path_length_and_global_efficiency(populations,projections)

    network = create_networkx_graph(populations,projections,prune=0)
    #v,l= network_vulnerability(network)


