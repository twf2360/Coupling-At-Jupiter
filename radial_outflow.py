'''
trying to recreate figure 6 within delbag (plasma radial density)
'''

import math
import numpy as np
import sys
import matplotlib.pyplot as plt
import json
from helpful_functions import HelpfulFunctions
#Magnitude of Jupiter's Magnetic Field at equator in T
B0 = 4.17 * 10 **-4

#Jupiter Equatorial Radius in M
Rj = 7.14 * 10 ** 7  #from Dessler's appendices 

mu_0 = 1.25663706212 * 10 ** -6


class radialOutflow:
    '''
    Class that plots the radial outflow velocity, and the local alfven velocity to recreate figure 6 within Delamere and Bagenal's Solar wind interaction with Jupiter’s magnetosphere
    Contains a magnetic dipole class that defines a simplified, dipolar version of Jupiter's magnetic field, which is used to calculate the alfven velocity
    
    - User Functions & Inputs:
    avgIonMass must be stated upon initalisation. This should be given as average atomic mass, and is multiplied by the mass of a proton to get the average mass in kg
    Datapoints - this calculates the values to be plotted. Inputs:
                MinR_RJ = the starting radial distance in Rj
                MaxR_RJ = the upper limit of radial distance in Rj
                NumPoints = the number of points at which outflow velocity and alfven velocity should be calculated
                mdots = a list(or array) of the mass loading in kgs-1 

    
    
    '''
    def __init__(self,avgIonMass):
        ''' 
        input average ion mass in amu
        '''
        self.avgIonMass = avgIonMass * 1.67 * 10**-27


    def flow_velocity(self, r, mdot):
        ''' 
        inputs:
        r = radial distance in rj
        mdot = mass loading rate in kg/s
        

        returns radial flow velocity at point r. 
        '''
        
        #density
        n = self.radial_density(r)

        #height of plasma torus
        z = 4 * Rj
        #radial cross sectional area of plasma torus
        A = z * 2 * np.pi * r * Rj

        v =  mdot/(n*self.avgIonMass*A)
        return v

    def radial_density(self, r):
        '''
        Calculates the density at a given radius r (in rj) along the centrifugal equator 
        returns n in m^-3
        '''
        n = (3.2e8 * r**(-6.9) + 9.9*r**(-1.28)) * 1e6 #think it should be R
        return n

    def plotRadialDensity(self, numpoints = 1000, start = 5, end = 70, show = 'off'):
        densities = []
        radii = []
        for r in np.linspace(start, end, numpoints):
            densities.append(self.radial_density(r))
            radii.append(r)
        
        
        if show == 'on':
            fig, ax = plt.subplots(figsize = (8,5))
            ax.plot(radii, densities, label = 'Density $(m^{-3})$')
            ax.legend()
            ax.set(xlabel='Radius $(R_J)$', ylabel='Density ($m^3$)', title='Density Vs Radial Distsance')
            ax.yaxis.set_ticks_position('both')
            plt.yscale("log")
            plt.savefig('images-24-jan-update/radial_density_profile.png')
            plt.show()
        return radii, densities
        
    def plotRadialDensityTwoSegments(self, numpoints = 1000, start1 = 5, end1 = 20, start2 =50, end2 =70):
        densities1 = []
        radii1 = []
        densities2 = []
        radii2 = []
        for r in np.linspace(start1, end1, numpoints):
            densities1.append(self.radial_density(r))
            radii1.append(r)
        
        for r in np.linspace(start2, end2, numpoints):
            densities2.append(self.radial_density(r))
            radii2.append(r)
        
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.semilogy(radii1, densities1, label = 'Density Section 1') #change to ax.plot to remove log scale
        ax2.semilogy(radii2, densities2, label = 'Density Section 2', color = 'r') #change to ax.plot ^^
        ax1.legend()
        ax1.set(ylabel='Density ($m^3$)' ,title='Density Vs Radial Distance')
        ax1.yaxis.set_ticks_position('both')
        ax2.legend()
        ax2.set(xlabel='Radius $(R_J)$', ylabel='Density ($m^{-3}$)') #, title='Density Vs Radial Distance')
        ax2.yaxis.set_ticks_position('both')
        plt.savefig('images-24-jan-update/radial_density_profile_two_points.png')
        plt.show()
        

        
    def local_Alfven_vel(self, r, theta = np.pi/2, phi = 0):
        help = HelpfulFunctions()
        ''' 
        inputs:
        r = radial distance from planet in rj

        returns alfven velocity in m/s at point r 
        '''
        magB = B0 * (1/r)**3
        n = self.radial_density(r)
        rho = self.avgIonMass * n
        denom = np.sqrt((mu_0 * rho))
        Va = magB/ denom
        #print('va = {}, rho = {}, magB = {}'.format(Va, rho, magB))
        #print("magB = {}, denom = {} \n n = {}, rho = {}, mu = {}, mass = {} va = {} \n \n".format(magB, denom, n, rho, mu_0, self.avgIonMass, Va))
        return Va

    def datapoints(self, minR_RJ, maxR_RJ, numpoints, mdots):
        minR = minR_RJ * Rj
        maxR = maxR_RJ * Rj
        r_values = np.linspace(minR_RJ, maxR_RJ, numpoints)
        alfven_vel_values = []
        flow_values = dict()

        i = 0
        for mdot in mdots:
            flow_for_given_mdot = []
            for r in r_values:
                if i == 0:    
                    Va = self.local_Alfven_vel(r)
                    alfven_vel_values.append(Va)
                    
                v = self.flow_velocity(r, mdot)
                flow_for_given_mdot.append(v)
            flow_values[mdot] = flow_for_given_mdot
            i = 1
        

        np.save("data/radial_flow/local_alfven.npy", alfven_vel_values, allow_pickle=True)
        np.save("data/radial_flow/flow_velocity.npy", flow_values, allow_pickle=True)
        myjson = json.dumps(flow_values)
        f = open("data/radial_flow/flow_values_dict.json", "w")
        f.write(myjson)
        f.close()
        np.save("data/radial_flow/r_values.npy", r_values, allow_pickle=True)
    

    def plotOutflow(self):
        '''
        requires there to be data already - plots outflow velocity against radial distance
        '''
        va_values = np.load("data/radial_flow/local_alfven.npy", allow_pickle=True)
        r_values = np.load("data/radial_flow/r_values.npy", allow_pickle=True)
        Rj_values = r_values/Rj
        #v_values = np.load("data/radial_flow/flow_velocity.npy", allow_pickle=True)
        with open("data/radial_flow/flow_values_dict.json", "r") as json_dict:
            v_values = json.load(json_dict)
        
        fig, ax = plt.subplots()
        ax.set(xlabel = 'Radial Distance (RJ)', ylabel = 'V $(kms^{-1})$')

       # print(type(v_values))
        #print(v_values)
        
        for key in v_values:
            v_kms = np.array(v_values[key]) /1000
            ax.plot(r_values, v_kms, label = 'Radial Velocity ({} = {}Kg/s)'.format(u'\u1E41' ,key))
            
        #print(va_values)
        va_kms = np.array(va_values)/1000
        ax.plot(r_values, va_kms, label = 'Local Alfven Velocity')
        ax.legend()
        box = ax.get_position()
        #ax.set_position([box.x0, box.y0 + box.height * 0.1,
        #                box.width, box.height * 0.9])

        # Put a legend below current axis
        #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                #fancybox=True, shadow=True, ncol=5)
        ax.legend()
        ax.yaxis.set_ticks_position('both')
        plt.yscale("log")
        plt.savefig('images-24-jan-update/radial_flow_plot.png')
        plt.show()

'''
radial = radialOutflow(28)

radial.datapoints(10, 100, 200, [280, 500, 1300])
radial.plotOutflow()
radial.plotRadialDensity(show='on')
radial.plotRadialDensityTwoSegments() #find a way to overlay the plots on top of each other?

'''

