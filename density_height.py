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
from field_and_current_sheet import InternalAndCS
Rj = 7.14 * 10 ** 7
class DensityHeight:
    def __init__(self, numpoints, start, stop):
        self.radialOutflowFunctions = radialOutflow(28)
        self.numpoints = numpoints
        self.help = HelpfulFunctions()
        self.start = start
        self.stop = stop 
        self.tracing = InternalAndCS()
    def scaleheight(self, R):
        a1 = -0.116
        a2 = 2.14
        a3 = -2.05
        a4 = 0.491
        a5 = 0.126
        r = np.log10(R/6) #<-- this is what we need to check 
        h = a1 + a2 * r + a3 * r**2 + a4 * r**3 + a5 * r**5
        H = 10**h #<--- this too 
        return H

    def density(self, n_0, z, H):
        n = n_0 * np.exp(-z/H)
        return n

    def plotting(self, density = 'on',scale_height = 'off'):
        radii, n_0s = self.radialOutflowFunctions.plotRadialDensity(start=self.start, end = self.stop, numpoints=self.numpoints)
        zs =  np.linspace(self.start, self.stop, self.numpoints)
        ns = []
        H_rj_s = []
        Hs = []
        if density == 'on':
            for z in zs:
                n_row = []
                for i in range(len(radii)):
                    H = self.scaleheight(radii[i]) 
                    #print(H)
                    n_0 = n_0s[i]
                    n = self.density(n_0, z, H)
                    n_row.append(n)
                ns.append(n_row)
            #print(ns)
            
            density_0_cm = np.array(n_0s)/ (10**6)
            density_cm = np.array(ns)/(10**6)
            fig, (ax1, ax2) = plt.subplots(1,2, figsize =(25,13))
            
            cont = ax2.contourf(radii, zs, density_cm, cmap = 'bone', locator=ticker.LogLocator())

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
                Hs.append(self.scaleheight(r))
            fig, ax = plt.subplots(figsize =(25,13))
            ax.plot(radii, Hs, label = 'Scale Height', color = 'g')
            #plt.xscale('log')
            ax.set(xlabel='Radius (RJ)', ylabel='Scale Height ($R_J$)', title='Scale height depenence on radial distance')
            plt.xlim(0, 100)
            plt.show()
    
    def equators(self):
        r = 30
        theta = np.pi/2
        phi_LH =  21* np.pi/180
        phi_rh = 2 *np.pi - phi_LH
        Btheta_eq = self.tracing.find_mag_equator(point=[r*Rj, theta,phi_LH])

        centrifugaleq_theta_lat = self.help.centrifugal_equator(r, phi_rh)# +np.pi)
        print(Btheta_eq, centrifugaleq_theta_lat)
        '''
        centrifugaleq_theta = np.pi/2 - centrifugaleq_theta_lat
        if centrifugaleq_theta > np.pi/2:
            diff = centrifugaleq_theta - np.pi/2
            centrifugaleq_theta -= 2*diff
        '''
        #print(centrifugaleq_theta, Btheta_eq)
        #print(centrifugaleq_LH_above_spin, centrifugaleq_RH_above_spin)

        centrifual_eq = np.array([[-r * np.cos(centrifugaleq_theta_lat), - r * np.sin(centrifugaleq_theta_lat)], [r * np.cos(centrifugaleq_theta_lat),  r * np.sin(centrifugaleq_theta_lat)]])
        b_eq = np.array([[-r * np.sin(Btheta_eq), - r * np.cos(Btheta_eq)], [r * np.sin(Btheta_eq),  r * np.cos(Btheta_eq)]])
        spin_eq = np.array([[-r,0],[r,0]])

        centrifual_eq_t = np.transpose(centrifual_eq)
        b_eq_t = np.transpose(b_eq)
        spin_eq_t = np.transpose(spin_eq)

        fig, ax = plt.subplots()
        
        ax.plot(centrifual_eq_t[0] , centrifual_eq_t[1], color = 'c', label = 'Centrifugal Equator')
        ax.plot(spin_eq_t[0], spin_eq_t[1], color = 'm', label = 'Spin Equator')
        ax.plot(b_eq_t[0], b_eq_t[1], color = 'k', label = 'Magnetic Field Equator')

        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        phi_lh_deg = phi_LH * 180/np.pi 
        ax.set(xlim = (-30,30), ylim = (-5,5), xlabel = 'X $R_J$', ylabel = 'Y $R_J$', title = 'Different Equators at Jupiter, SYSIII longitude =   {:.0f} deg on RHS'.format(phi_lh_deg))
        ax.set_aspect(aspect='equal')
        plt.legend()
        plt.show()

    
        


    


    

'''
test = DensityHeight(numpoints= 100, start= 5, stop = 20)
<<<<<<< HEAD
#test.plotting(scale_height='off', density = 'on')    
test.equators()
=======
test.plotting(scale_height='on', density = 'on')    
>>>>>>> topdown

