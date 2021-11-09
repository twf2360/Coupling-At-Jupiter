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
B0 = 4.17e-7

#Jupiter Equatorial Radius in M
Rj = 7.14e7 #from Dessler's appendices 

mu_0 = 4* 10e-7* math.pi


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
        
        field = radialOutflow.magneticDipole(B0)
        self.field = field 
        self.avgIonMass = avgIonMass * 1.67e-27

    class magneticDipole:
        ''' 
        define the magnetic dipole 
        '''
        def __init__(self, strength):
            self.strength = strength
        
        def cart_to_sph(self, x, y ,z):
            '''
            turn the cartesian co-ordinates of a point into spherical 
            '''

            r = math.sqrt(x**2 + y**2 + z**2)
            theta = math.atan2(math.sqrt(x**2 + y**2), z)
            phi = math.atan2(x,y)
            return r, theta, phi

        def sph_to_cart(self, r, theta, phi):
            x = r*np.sin(theta)*np.cos(phi)
            y = r*np.sin(theta)*np.sin(phi)
            z = r*np.cos(theta)
            return x,y,z
        
        def field_at_point(self, cordinates = np.array([0,0,0], dtype = float), coord_type = 'sph'):
            if coord_type == 'cart':
                x = cordinates[0]
                y = cordinates[1]
                z = cordinates[2]
                r, theta, phi = self.cart_to_sph(x, y, z)
            elif coord_type == 'sph':
                r = cordinates[0]
                theta = float(cordinates[1])
                phi = cordinates[2]
            else:
                print('co-ordinate type not recognised.')
                sys.exit()
            
            #Overall strength of the vector will scale with distance 
            ScaleFactor = self.strength * (Rj/r)**3

            #magnetic field in radial and polar direction
            
            B_r = -2 * ScaleFactor * np.cos(theta)
            B_theta = - ScaleFactor * np.sin(theta)
            B_phi = 0

            return B_r, B_theta, B_phi

        
        
    
    def flow_velocity(self, r, mdot):
        ''' 
        inputs:
        r = radial distance 
        mdot = mass loading rate 
        m = average ion mass

        returns radial flow velocity at point r. 
        '''
        
        #density
        n = self.radial_density(r)

        #height of plasma torus
        z = 4 * Rj
        #radial cross sectional area of plasma torus
        A = z * 2 * np.pi * r

        v =  mdot/(n*self.avgIonMass*A)
        return v

    def radial_density(self, r):
        
        R = r/Rj
        n = 3.2e8 * R**(-6.9) + 9.9*R**(-1.28) * 1e6
        return n

    def local_Alfven_vel(self, r, theta = np.pi/2, phi = '0'):
        help = HelpfulFunctions()
        ''' 
        inputs:
        r = radial distance from planet 

        returns alfven velocity at point r 
        '''
        Br, Btheta, Bphi = self.field.field_at_point([r,theta,phi])
        B = help.Bsph_to_Bcart(Br, Btheta, Bphi, r, theta, phi)
        n = self.radial_density(r)
        rho = self.avgIonMass * n
        magB = np.linalg.norm(B)
        Va = magB/(mu_0 * rho)
        print("magB = {}, n = {}, rho = {}, mu = {}, mass = {} va = {}".format(magB, n, rho, mu_0, self.avgIonMass, Va))
        return Va

    def datapoints(self, minR_RJ, maxR_RJ, numpoints, mdots):
        minR = minR_RJ * Rj
        maxR = maxR_RJ * Rj
        r_values = np.linspace(minR, maxR, numpoints)
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
    

    def plot(self):
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
        ax.set(xlabel = 'Radial Distance (RJ)', ylabel = 'v(kms)')
       # print(type(v_values))
        #print(v_values)
        
        for key in v_values:
            v_kms = np.array(v_values[key]) /1000
            ax.plot(Rj_values, v_kms, label = 'mdot ={}'.format(key))
            
        #print(va_values)
        va_kms = np.array(va_values)/1000
        #ax.plot(Rj_values, va_kms, label = 'local alfven velocity')
        ax.legend()
        plt.show()
        plt.savefig('images/radial_flow_plot.png')

        
radial = radialOutflow(28)
radial.datapoints(1, 100, 200, [280, 500, 1300])
radial.plot()
 




