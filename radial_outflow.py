'''
trying to recreate figure 6 within delbag (plasma radial density)
'''

import math
import numpy as np
import sys
import matplotlib.pyplot as plt

#Magnitude of Jupiter's Magnetic Field at equator in T
B0 = 4.17e-5

#Jupiter Equatorial Radius in M
Rj = 7.14e7 #from Dessler's appendices 

mu_0 = 4e-7* math.pi


class radialOutflow:

    def __init__(self,avgIonMass):
        self.field = magneticDipole(B0)
        self.avgIonMass = avgIonMass
    
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
        A = z * 2 * pi * r

        v =  mdot/(n*self.avgIonMass*a)
        return v

    def radial_density(self, r):
        ''' 
        the radial ion density function as from Frank and Patterson: https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2001JA000077
        '''
        R = r/Rj
        if R > 20:
            n = 3.2e8 * R**(-6.90)
        if R < 50:
            n = 9.8 * R*(-1.28)
        if 20<R<50:
            n = 'something'
        return n

    def local_Alfven_vel(self, r, theta = '0', phi = '0'):
        ''' 
        inputs:
        r = radial distance from planet 

        returns alfven velocity at point r 
        '''
        B = self.field.field_at_point([r,theta,phi])
        n = self.radial_density(r)
        rho = self.avgIonMass * n
        magB = np.linalg.norm(B)
        Va = magB/(mu_0 * rho)
        return Va

    def datapoints(self, minR, maxR, numpoints, mdots):
        r_values = np.linspace(minR, maxR, numpoints)
        alfven_vel_values = []
        flow_values = dict()
        
        if not type(mdots) == tuple:
            print('please enter mdot values as a tuple ie. surrounded by brackets with values seperated by commas')
            sys.exit()
        
        for mdot in mdots:
            i = 0
            flow_for_given_mdot = []
            for r in r_values:
                if i == 0:    
                    Va = self.local_Alfven_vel(r)
                    alfven_vel_values.append(Va)
                    i = 1 
                v = self.flow_velocity(r, mdot)
                flow_for_given_mdot.append(v)
            flow_values[mdot] = flow_for_given_mdot

        np.save("data/radial_flow/local_alfven.npy", alfven_vel_values, allow_pickle=True)
        np.save("data/radial_flow/flow_velocity.npy", flow_values, allow_pickle=True)
        np.save("data/radial_flow/r_values.npy", r_values, allow_pickle=True)


def testplot():
    '''
    requires there to be saved data already - plots outflow velocity against radial distance
    '''
    va_values = np.load("data/radial_flow/local_alfven.npy")
    r_values = np.load("data/radial_flow/r_values.npy")
    Rj_values = r_values/Rj
    v_values = np.load("data/radial_flow/flow_velocity.npy")
    fig, ax = plt.subplots()
    ax.set(xlabel = 'Radial Distance (RJ)', ylabel = 'v')
    for key in v_values:
        ax.plot(Rj_values, v_values[key], label = 'mdot ='.format(key))
    ax.plot(Rj_values, va_values, label = 'local alfven velocity')
    plt.show()
    plt.savefig('images/radial_flow_plot.png')

        
            
          






class magneticDipole:
    ''' 
    define the magnetic dipole 
    '''
    def __init__(strength):
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
            theta = cordinates[1]
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

    def field_at_point_in_cart(self,cordinates = np.array([0,0,0], dtype = float), coord_type = 'sph'):
        B_r, B_theta, B_phi = self.field_at_point(cordinates, coord_type)
        B_x, B_y, B_z = self.sph_to_cart(B_r, B_theta, B_phi)
