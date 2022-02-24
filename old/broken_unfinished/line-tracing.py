from mag_field_models import field_models
import numpy as np
import matplotlib.pyplot as plt 
import math
import scipy.special
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d
from helpful_functions import HelpfulFunctions
Rj = 7.14e7 
''' 
idea 1 - use quiver to plot the 3d vectors. 

'''


class linetrace():
    def __init__(self, NumPoints, gridsize):
        self.NumPoints = NumPoints
        self.gridsize = gridsize
    
    def makegrid(self):
        points_min = -self.gridsize/2
        points_max = self.gridsize/2
        x,y,z = np.meshgrid(np.linspace(points_min, points_max, self.NumPoints),
                            np.linspace(points_min, points_max, self.NumPoints),
                            np.linspace(points_min, points_max, self.NumPoints))
        return x,y,z

    def getdata(self):
        gridx,gridy,gridz = self.makegrid()


        ''' 
        it seems to be the case with quiver that you can just define what the vectors are, instead of having to calculate the values at all sorts of different points and then plot 
        those calculated values.
        however, i dont know the best way to go about doing that 
        '''
        x_vector_grid = []
        y_vector_grid = []
        z_vector_grid = []
        field = field_models()
        for k in range(self.NumPoints):
            print('k={}'.format(k))
            x_vector_column = []
            y_vector_column = []
            z_vector_column = []
            for i in range(self.NumPoints):
                #print(i)
                x_vector_row = []
                y_vector_row = []
                z_vector_row = []
                for j in range(self.NumPoints):
                    x = gridx[k][i][j]
                    #print(x)
                    y = gridy[k][i][j]
                    z = gridz[k][i][j]
                    r = math.sqrt(x**2 + y**2 +z**2)
                    theta = np.arctan2(np.sqrt(x**2+y**2),z)
                    phi = np.arctan2(y,x)
                    Br, Btheta, Bphi = field.Internal_Field(r, theta, phi, model='VIP4')
                    Bx = Br * np.sin(theta) * np.cos(phi) + Btheta * np.cos(theta) * np.cos(phi) - Bphi *np.sin(phi)
                    By = Br * np.sin(theta) * np.sin(phi) + Btheta * np.cos(theta) * np.sin(phi) + Bphi *np.cos(phi)
                    Bz = Br * np.cos(theta)  - Btheta * np.sin(theta) 
                    x_vector_row.append(Bx)
                    y_vector_row.append(By)
                    z_vector_row.append(Bz)
                x_vector_column.append(x_vector_row)
                y_vector_column.append(y_vector_row)
                z_vector_column.append(z_vector_row)
            x_vector_grid.append(x_vector_column)
            y_vector_grid.append(x_vector_column)
            z_vector_grid.append(x_vector_column)

        grid = [gridx, gridy, gridz]
        np.save("data/line-tracing/grid.npy", grid)
        np.save("data/line-tracing/x_vectors.npy", x_vector_grid)
        np.save("data/line-tracing/y_vectors.npy", y_vector_grid)
        np.save("data/line-tracing/z_vectors.npy", y_vector_grid)

    def makeplot(self):

        grid = np.load("data/line-tracing/grid.npy")
        gridx = grid[0]
        gridy = grid[1]
        gridz = grid[2]
        x_vector_grid = np.load("data/line-tracing/x_vectors.npy")
        y_vector_grid = np.load("data/line-tracing/y_vectors.npy")
        z_vector_grid = np.load("data/line-tracing/z_vectors.npy")

        test_x_vector = 1e23 * np.array(x_vector_grid)
        test_y_vector = 1e23* np.array(y_vector_grid)
        test_z_vector = 1e23* np.array(z_vector_grid)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        #print(x_vector_grid)
        #print(gridx)
        #print(test_x_vector)
        ax.quiver(gridx, gridy, gridz, test_x_vector, test_y_vector, test_z_vector, color = 'black', zorder = 0)
        #ax.quiver(x, y, z, x_vector_grid, y_vector_grid, z_vector_grid, color = 'black', normalize = True, arrow_length_ratio = 1)

        #adding the cirlce that is Jupiter:
        N=50
        stride=2
        u = np.linspace(0, 2 * np.pi, N)
        v = np.linspace(0, np.pi, N)
        xcircle = Rj * np.outer(np.cos(u), np.sin(v))
        ycircle = Rj * np.outer(np.sin(u), np.sin(v))
        zcircle = Rj * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(xcircle, ycircle, zcircle, linewidth=0.0, cstride=stride, rstride=stride, color = 'yellow', zorder=100)
        

        plt.show()


linetrace = linetrace(50, 60e7)
linetrace.makeplot()



''' 
idea 2: make a test particle, and then plot it's motion in the magnetic field.
'''
def pathtrace(r,theta,phi, iterations = 10000, timestep=100):
    ''' 
    define the initial starting co-ordinates of a point for which the line is to be traced
    '''
    
    field = field_models()
    for i in iterations:
        Br, Bint, Bphi = field.Internal_Field(r, theta, phi, model='VIP4')

#def motion(q, E, v, B):


