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
from matplotlib import ticker, cm

Rj = 7.14 * 10 ** 7
class DensityHeight:
    def __init__(self, numpoints, start, stop):
        self.radialOutflowFunctions = radialOutflow(28)
        self.numpoints = numpoints
        self.help = HelpfulFunctions()
        self.start = start
        self.stop = stop 

    def scaleheight(self, R):
        a1 = -0.116
        a2 = 2.14
        a3 = -2.05
        a4 = 0.491
        a5 = 0.126
        r = np.log10(R/(6*Rj)) #<-- this is what we need to check 
        h = a1 + a2 * r + a3 * r**2 + a4 * r**3 + a5 * r**5
        H = 10**h #<--- this too 
        return H

    def density(self, n_0, z, H):
        n = n_0 * np.exp(-z/H)
        return n

    def plotting(self, density = 'on',scale_height = 'off'):
        radii, n_0s = self.radialOutflowFunctions.plotRadialDensity(start=self.start*Rj, end = self.stop*Rj, numpoints=self.numpoints)
        zs =  np.linspace(self.start*Rj, self.stop *Rj, self.numpoints)
        zs_rj = zs/Rj
        ns = []
        H_rj_s = []
        if density == 'on':
            for z in zs:
                n_row = []
                for i in range(len(radii)):
                    H = self.scaleheight(radii[i]*Rj) *Rj
                    #print(H)
                    n_0 = n_0s[i]
                    n = self.density(n_0, z, H)
                    n_row.append(n)
                ns.append(n_row)
            #print(ns)
            
            density_0_cm = np.array(n_0s)/ (10**6)
            density_cm = np.array(ns)/(10**6)
            fig, (ax1, ax2) = plt.subplots(1,2, figsize =(25,13))
            
            cont = ax2.contourf(radii, zs_rj, density_cm, cmap = 'bone', locator=ticker.LogLocator())

            ax2.set(xlabel = 'Radial Distance($R_J$)', ylabel = 'Height($R_J$)', title = 'Contour plot of density depending on radial density and height')
            ax2.yaxis.set_ticks_position('both')
            plt.colorbar(cont, label = 'Density ($cm^{-3}$)')
            
            ax1.plot(radii, density_0_cm, label = '$n_0$')
            ax1.legend()
            
            ax1.set(xlabel='Radius (RJ)', ylabel='Density ($cm^{-3}$)', title='Density Vs Radial Distsance at height = 0')
            ax1.yaxis.set_ticks_position('both')
            ax1.set_yscale("log")
            plt.suptitle('Density Variations as a function of height and radial distance ')
            #fig.tight_layout()
            plt.show()
        
        if scale_height == 'on':
            for r in radii:
                H_rj_s.append(self.scaleheight(r * Rj))
            fig, ax = plt.subplots(figsize =(25,13))
            ax.plot(radii, H_rj_s, label = 'Scale Height', color = 'g')
            plt.xscale('log')
            ax.set(xlabel='Radius (RJ)', ylabel='Scale Height ($R_J$)', title='Scale height depenence on radial distance')
            plt.xlim(0, 100)
            plt.show()

    
'''

test = DensityHeight(numpoints= 100, start= 5, stop = 20)
test.plotting(scale_height='off', density = 'on')    

'''