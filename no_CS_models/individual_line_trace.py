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
#mpl.rcParams['agg.path.chunksize'] = 10000
mpl.rcParams['legend.fontsize'] = 10

Rj = 7.14e7
class individualFieldTrace:
    ''' 

    TO DO - CHECK IF IT WORKS, PLOT THE TRACE, EXTEND TO MORE COMPLICATED MODELS? COULD DO THE 463 MODEL WHICH WILL INCLUDE A BPHI COMPONENT DUE TO SPINNING PLANET 

    '''

    def __init__(self, starting_cordinates, coord_type = "sph", equatorial_strength = 4.17e-7, model = 'dipole'):
        self.help = HelpfulFunctions()
        if model == 'dipole':
            self.field = self.magneticDipole(equatorial_strength)
        if not (coord_type == 'sph' or coord_type == 'cart'):
            print('please input co-ordinates in spherical polar or cartesian ("sph" / "cart") \n I will extend to further co-ordinate types later')
        if coord_type == 'cart':
            x,y,z = starting_cordinates[0], starting_cordinates[1], starting_cordinates[2]
            r,theta,phi = self.help.cart_to_sph(x, y, z)
            self.starting_cordinates = [r,theta,phi]
        if coord_type == "sph":
            self.starting_cordinates = starting_cordinates

    def trace_lower_hemisphere(self, printing = 'off'):
        coordinates = np.array(self.starting_cordinates)
        points = []
        i = 0
        while True:
            i += 1
            px, py, pz = self.help.sph_to_cart(coordinates[0],coordinates[1],coordinates[2])
            points.append([px,py,pz])
            r = coordinates[0]
            if r <= 3 * Rj:
                break


            B_r, B_theta, B_phi = self.field.field_at_point(cordinates = coordinates)
            B_x, B_y, B_z = self.help.Bsph_to_Bcart(B_r, B_theta, B_phi, coordinates[0], coordinates[1],coordinates[2])# quad =1)
            if not coordinates[2]  == 0:
                print('PHI NOT EQUAL 0')
                sys.exit()
            B = np.array([B_x, B_y, B_z])
            coordinates = [px,py,pz]
            Bunit = self.help.unit_vector_cart(B)
            dr = r * 0.0001 #(*Rj) #THIS IS HOW WE UPPDATE THE COORDINATES - IF IT TAKES TOO LONG, THIS NEEDS CHANGING IF IT TAKES TOO LONG OR IS GETTING WEIRD CLOSE TO PLANET
            change = dr * Bunit
            coordinates = np.add(coordinates, change)
            pr, ptheta, pphi = self.help.cart_to_sph(coordinates[0], coordinates[1], coordinates[2])
            coordinates = [pr,ptheta,pphi]

            
            if printing == 'on':
                if (i % 1000) == 0 or i == 1:
                    print('B cartesian = {}, B sph = [{} {} {}]'.format(B, B_r, B_theta, B_phi))
                    print('r = {}, theta = {}, phi = {}'.format(coordinates[0],coordinates[1],coordinates[2]))
                    print(' x= {}, y = {}, z =  {}'.format(px,py,pz))
                    print('bunit = {}, change = {}, dr = {} \n \n'.format(Bunit, change, dr))
            
            if i % 1000 == 0:
               print("theta = {}".format(coordinates[1]))

        return points
        

    def trace_upper_hemisphere(self, printing='off'):
        coordinates = np.array(self.starting_cordinates)
        points = []
        i = 0
        while True:
            i += 1           
            px, py, pz = self.help.sph_to_cart(coordinates[0],coordinates[1],coordinates[2])

            points.append([px,py,pz])
            
            r = coordinates[0]
            if r <= 3 * Rj:
                break


            B_r, B_theta, B_phi = self.field.field_at_point(coordinates)
            B_x, B_y, B_z = self.help.Bsph_to_Bcart(B_r, B_theta, B_phi, coordinates[0], coordinates[1],coordinates[2])
            B = np.array([B_x, B_y, B_z])
            coordinates = [px,py,pz]
            Bunit = self.help.unit_vector_cart(B)
            dr = r * 0.0001 #(*Rj) #THIS IS HOW WE UPPDATE THE COORDINATES - IF IT TAKES TOO LONG, THIS NEEDS CHANGING IF IT TAKES TOO LONG OR IS GETTING WEIRD CLOSE TO PLANET
            change = dr * - Bunit
            coordinates = np.add(coordinates, change)
            pr, ptheta, pphi = self.help.cart_to_sph(coordinates[0], coordinates[1], coordinates[2])
            coordinates = [pr,ptheta,pphi]
            if printing == 'on':
                if (i % 1000) == 0 or i == 1:
                    print('B cartesian = {}, B sph = [{} {} {}]'.format(B, B_r, B_theta, B_phi))
                    print('r = {}, theta = {}, phi = {}'.format(coordinates[0],coordinates[1],coordinates[2]))
                    print(' x= {}, y = {}, z =  {}'.format(px,py,pz))
                    print('bunit = {}, change = {}, dr = {} \n \n'.format(Bunit, change, dr))

        return points

    def plotTrace(self):
        lower = np.array(self.trace_lower_hemisphere())
        upper = np.array(self.trace_upper_hemisphere())

        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plottable_lists_upper = np.transpose(upper)
        plottable_lists_lower = np.transpose(lower)

        ax.plot(plottable_lists_upper[0], plottable_lists_upper[1], plottable_lists_upper[2],color = 'black', label = 'Field Trace')
        ax.plot(plottable_lists_lower[0], plottable_lists_lower[1], plottable_lists_lower[2], color = 'black', label = 'Field Trace')
        #make the sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = Rj * np.outer(np.cos(u), np.sin(v))
        y = Rj * np.outer(np.sin(u), np.sin(v))
        z = Rj * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color = 'yellow', zorder=100, label = 'Jupiter')
        ax.set_xlim3d(-40*Rj, 40*Rj)
        ax.set_ylim3d(-40*Rj, 40*Rj)
        ax.set_zlim3d(-40*Rj, 40*Rj)
        ax.set_xlabel('$X$', fontsize=10)
        ax.set_ylabel('$Y$', fontsize=10)
        #ax.legend()
        plt.savefig('images/individual_mag_field_trace.png')
        plt.show()

    def plot2d_dipole(self):
        '''
        will only see sensible results if y = 0 throughout and for dipole field 
        plots the x-z plane 
        '''

        lower = np.array(self.trace_lower_hemisphere())
        upper = np.array(self.trace_upper_hemisphere())

        
        fig, ax = plt.subplots()
        plottable_lists_upper = np.transpose(upper)
        plottable_lists_lower = np.transpose(lower)

        ax.plot(plottable_lists_upper[0], plottable_lists_upper[2],color = 'black', label = 'Field Trace')
        ax.plot(plottable_lists_lower[0], plottable_lists_lower[2], color = 'black')
        #make the circle
        ax.add_patch(Circle((0,0), Rj, color='y', zorder=100, label = "Jupiter"))
        ax.legend()
        plt.savefig('images/individual_mag_field_trace_2d.png')
        plt.show()



    class magneticDipole:
        ''' 
        define the magnetic dipole 
        '''
        def __init__(self, strength):
            self.strength = strength

        def field_at_point(self, cordinates = np.array([0,0,0], dtype = float), coord_type = 'sph'):
            helpful = HelpfulFunctions()
            if coord_type == 'cart':
                x = cordinates[0]
                y = cordinates[1]
                z = cordinates[2]
                r, theta, phi = helpful.cart_to_sph(x, y, z)
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
            
            B_r =  2 * ScaleFactor * np.cos(theta)
            B_theta = ScaleFactor * np.sin(theta)
            B_phi = 0

            return B_r, B_theta, B_phi  


test = individualFieldTrace([30*Rj, np.pi/2, 0])
test.plot2d_dipole()