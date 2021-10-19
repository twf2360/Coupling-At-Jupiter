import sys
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import axes3d
Mj = 1.898e27 #Kg
Rj = 7.14e7 #m
Tj = 32456 #s
B0 = 4.17e-5 #T
MagMomj = 2.83e20 #T·m3


def makegrid(NumPoints, gridsize):
    points_min = -gridsize/2
    points_max = gridsize/2
    x,y,z = np.meshgrid(np.linspace(points_min, points_max, NumPoints),
                        np.linspace(points_min, points_max, NumPoints),
                        np.linspace(points_min, points_max, NumPoints))
    return x,y,z



def makeplot(NumPoints, gridsize, moment):
    x,y,z = makegrid(NumPoints, gridsize)
    

    i = (3 * moment * x * z)/ ((x**2 + y**2 + z**2)**(5/2))
    j = (3 * moment * y * z)/ ((x**2 + y**2 + z**2)**(5/2))
    k = moment * (3*z**2 - (x**2 + y**2 + z**2))/((x**2 + y**2 + z**2)**(5/2))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.quiver(x, y, z, i, j, k, color = 'black')

    plt.show()

makeplot(50, 60e7, MagMomj)

'''
not sure why this doesnt seem to work? 
'''