import math
import numpy as np
import sys
import matplotlib.pyplot as plt
import json
from helpful_functions import HelpfulFunctions
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib as mpl
from mag_field_models import field_models
import scipy
import scipy.special
from radial_outflow import radialOutflow


Rj = 7.14 * 10 ** 7
class DensityHeight:
    def __init__(self, numpoints):
        self.radialOutflowFunctions = radialOutflow(28)
        self.numpoints = numpoints
        self.help = HelpfulFunctions()

    def scaleheight(self, r):
        a1 = -0.116
        a2 = 2.14
        a3 = -2.05
        a4 = 0.491
        a5 = 0.126
        R = np.log10(r/(6*Rj))
        h = a1 + a2 * R + a3 * R**2 + a4 * R**3 + a5 * R**5
        H = 10**h
        return h

    def density(self, n_0, z, H):
        n = n_0 * np.exp(-z/H)
        return n

    def plotting(self):
        radii, n_0s = self.radialOutflowFunctions.plotRadialDensity(start=5*Rj, end = 75*Rj, numpoints=self.numpoints)

        zs =  np.linspace(5*Rj, 75 *Rj, self.numpoints)
        zs_rj = zs/Rj
        ns = []
        for z in zs:
            n_row = []
            for i in range(len(radii)):
                H = self.scaleheight(radii[i]*Rj)
                n_0 = n_0s[i]
                n = self.density(n_0, z, H)
                n_row.append(n)
            ns.append(n_row)
        print(ns)
        plt.contour(radii, zs_rj, ns, 20, cmap = 'RdGy')
        plt.show()


test = DensityHeight(100)
test.plotting()    


    
'''
    def scale_heights(self):
        radii, n_0s = self.radialOutflowFunctions.plotRadialDensity(numpoints=self.numpoints, start=5*Rj, end= 55*Rj)
        scale_heights = []
        for r in radii:
            H = self.scaleheight(r)
            scale_heights.append(H)
        return radii, n_0s, scale_heights
    
    def density(self, n_0, z, H):
        n = n_0 * np.exp(-z/H)
        return n
    
    def plotting(self):
        radii, n_0s, scale_heights = self.scale_heights()
        zs = np.linspace(5*Rj, 55 *Rj, self.numpoints)
        ns = []
        for i in range(len(zs)) :
            for j in range(len(radii)):
                n = self.density(n_0s[i], zs[i], scale_heights[])

test = DensityHeight(100)
test.plotting()

    
        

 
zs = np.linspace(0, 50 *Rj, self.numpoints)
        radii, n_0s, scale_heights = self.scale_heights()
        ns = []
        for i in range(self.numpoints):
            n = self.density(n_0s[i], zs[i], scale_heights[i])
            ns.append(n)

        plt.contour(scale_heights, radii,n, 20, cmap = 'RdGy')
        plt.show()

'''