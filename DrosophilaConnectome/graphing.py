import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from operator import itemgetter

def plot_matrix(data,lpus):
    """ create and plot a dense matrix from a sparse representations """
    m_size = len(lpus)
    m =  np.zeros((m_size,m_size))

    for j,pre in enumerate(lpus):
        for i,post in enumerate(lpus):
            try:
                m[i,j] = data[pre][post]
            except:
                pass

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    cax = ax.matshow(m, interpolation='nearest')
    fig.colorbar(cax)

    ax.set_xticklabels(['']+lpus)
    ax.set_yticklabels(['']+lpus)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()


def plot_measurement(data,xlabel= "X Data",ylabel="Y Data"):
    """ Plot a measurement bar graph """ 

    s = sorted(data.iteritems(), key=itemgetter(1), reverse=True)

    label = zip(*s)[0]
    score = zip(*s)[1]
    x_pos = np.arange(len(label)) 

    plt.bar(x_pos, score,align='center')
    plt.xticks(x_pos, label,rotation=90) 
    plt.ylabel(ylabel)
    plt.show()
