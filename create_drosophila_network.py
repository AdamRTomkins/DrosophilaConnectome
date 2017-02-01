"""
A script to convert the drosophila connectome into SpineML 
 
# Install libSpineML from source
# https://github.com/AdamRTomkins/libSpineML

"""
from libSpineML import smlExperiment as exp
from libSpineML import smlNetwork as net
from libSpineML import smlComponent as com

import csv
import sys
import ipdb

import cStringIO
import graphviz as gv
import matplotlib.pyplot as plt

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
                    g1.edge(pre, post,weight = str(len(projections[pre][post])))


    filename = g1.render(filename='left_hemisphere')
    print filename

if __name__ == "__main__":
    main()

