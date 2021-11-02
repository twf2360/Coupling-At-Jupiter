"""
Tracing a massively simplified version of Jupiter's magnetic dipole 

Author @twf2360
"""

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

#Magnitude of Jupiter's Magnetic Field at equator in T
B0 = 4.17e-5

#Jupiter Equatorial Radius in M
Rj = 7.14e7 #from Dessler's appendices 



def MagFieldVector_2D(x,y):
    '''
    return the magnetic field vector at a point (x,y), imagining a 2D slice 
    '''
   
    #defining the point in terms of spherical polar
    r = math.sqrt(x**2 + y**2)
    theta = math.atan2(y,x)
    
    #Overall strength of the vector will scale with distance 
    ScaleFactor = B0 * (Rj/r)**3

    #magnetic field in radial and polar direction
    B_r = -2 * ScaleFactor * np.cos(theta)
    B_theta = - ScaleFactor * np.sin(theta)

    #back into cartesian 
    B_x = -B_theta * np.sin((math.pi/2) + theta) + B_r *  np.cos((math.pi/2) + theta)
    B_y =  B_theta* np.cos((math.pi/2) + theta) + B_r * np.sin((math.pi/2) + theta)
    
    return B_x, B_y


def Trace(NumPoints, GridSize=60e7):
    '''
    Trace out the vectors calculated by MagFieldVector_2D, onto a grid defined by the number of points and the size of the grid 
    '''
    xmin, xmax = -GridSize/2, GridSize/2
    ymin, ymax = -GridSize/2, GridSize/2

    xpoints = np.linspace(xmin, xmax, NumPoints)
    ypoints = np.linspace(ymin, ymax, NumPoints)

    meshx, meshy = np.meshgrid(xpoints, ypoints)
    
    x_vector_grid = []
    y_vector_grid = []


    for i in range(len(meshx)):
        x_vector_row = []
        y_vector_row = []
        for j in range(len(meshx[0])):
            xvector, yvector = MagFieldVector_2D(meshx[i][j], meshy[i][j])
            x_vector_row.append(xvector)
            y_vector_row.append(yvector)
        x_vector_grid.append(x_vector_row)
        y_vector_grid.append(y_vector_row)  
  
    x_vector_grid = np.array( x_vector_grid)
    y_vector_grid = np.array(y_vector_grid)
    fig, ax = plt.subplots()
    ax.add_patch(Circle((0,0), Rj, color='y', zorder=100, label = "Jupiter"))
    colour = np.log(np.hypot(x_vector_grid, y_vector_grid))
    ax.streamplot(meshx, meshy, x_vector_grid, y_vector_grid, linewidth=1, color = colour, cmap=plt.cm.inferno,
           density=2, arrowstyle='->', arrowsize=1.5)
    l = plt.legend(loc='upper right')
    l.set_zorder(2000)  
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.savefig('images/first_mag_field_trace.png')    
    plt.show()
    

    
Trace(100)