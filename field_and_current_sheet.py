''' 
extends the individual line trace to trace any given field provided by the mmag field models class
'''

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
#mpl.rcParams['agg.path.chunksize'] = 10000
mpl.rcParams['legend.fontsize'] = 10

Rj = 7.14e7
class InternalAndCS:
    ''' 
    Class to plot the combined magnetic field; both internally generated the 
    field thanks to the current sheet
    
    '''

    def __init__(self, starting_cordinates = [30*Rj, np.pi/2, np.pi/2], coord_type = "sph", equatorial_strength = 4.17e-7, model = 'VIP4'):
        ''' 
        Plots the magnetic field of jupiter, both internally generated (as defined by input model) and due to the current sheet

        Inputs:
        starting_cordinates: the initial position from which the magnetic field should be calculated, with Jupiter being the origin
        coord_type = Whether the starting co-ordinates are spherical (system III) or cartesian
        equatorial_strength = the equatorial field strength on the surface of Jupiter in T
        model = the model for the internally generated field. Must be included within Chris's model



        '''
        self.help = HelpfulFunctions() #a list of helpful functions that come up repeatedly

        models = ['JRM09', 'VIP4', 'VIT4', 'Ulysses 17ev', 'V1-17ev','O6','O4', 'SHA', 'dipole']
        
        if model not in models:
            print('model not recognised')
            sys.exit()
        
        else:
            self.model = model
        self.field = field_models()

        if not (coord_type == 'sph' or coord_type == 'cart'):
            print('please input co-ordinates in spherical polar or cartesian ("sph" / "cart") \n I will extend to further co-ordinate types later')
        
        if coord_type == 'cart': #if the coordinates are entired in cartersian they are changed to be in spherical polar
            x,y,z = starting_cordinates[0], starting_cordinates[1], starting_cordinates[2]
            r,theta,phi = self.help.cart_to_sph(x, y, z)
            self.starting_cordinates = [r,theta,phi]
        
        if coord_type == "sph":
            self.starting_cordinates = starting_cordinates

    def trace_lower_hemisphere(self, printing = 'off', starting_cordinates = None):
        ''' 
        Trace the lower hemisphere of the magnetic field, starting from the initially defined co ordinated.
        If print = on, more sanity checks will be printed! 
        '''
        print('\n Lower Hemisphere')
        if starting_cordinates == None:
            starting_cordinates = self.starting_cordinates
        coordinates = starting_cordinates
        points = []
        i = 0
        
        while True:
            ''' 
            loops for as long as r is larger than a defined value
            '''
            i += 1 #this i is just used to define when to print things! 
            px, py, pz = self.help.sph_to_cart(coordinates[0],coordinates[1],coordinates[2])
            points.append([px,py,pz])
            r = coordinates[0]
            
            if r <= 3 * Rj: #defines when the loop is broken out of 
                break


            B_r, B_theta, B_phi = self.field.Internal_Field(r/Rj, coordinates[1], coordinates[2], model=self.model) #calculates the magnetic field due to the internal field in spherical polar that point
            if i % 1000 == 0:
                print(coordinates)
            B_current = self.field.CAN_sheet(r/Rj, coordinates[1], coordinates[2]) #calculates the magnetic field due to the current sheet in spherical polar
            B_notcurrent = np.array([B_r, B_theta, B_phi]) 
            B_overall = np.add(B_current, B_notcurrent) #adds up the total magnetic field 
            B_x, B_y, B_z = self.help.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], coordinates[0], coordinates[1],coordinates[2]) #converts magnetic field to cartesian
            B = np.array([B_x, B_y, B_z])
            coordinates = [px,py,pz] #change the definition of the coordinates from spherical to cartesian 
            Bunit = self.help.unit_vector_cart(B) #calculates the unit vector in cartesian direction
            dr = r * 0.001  #THIS IS HOW WE UPPDATE THE COORDINATES - IF IT TAKES TOO LONG, THIS NEEDS CHANGING IF IT TAKES TOO LONG OR IS GETTING WEIRD CLOSE TO PLANET
            change = dr * Bunit #the change from this coordinate to the next one is calculated
            coordinates = np.add(coordinates, change) #add the change to the current co ordinate
            pr, ptheta, pphi = self.help.cart_to_sph(coordinates[0], coordinates[1], coordinates[2]) #change the coordinatres back in spherical
            coordinates = [pr,ptheta,pphi] 

            
            if printing == 'on':
                if (i % 1000) == 0 or i == 1:
                    print('B cartesian = {}, B sph = [{} {} {}]'.format(B, B_r, B_theta, B_phi))
                    print('r = {}, theta = {}, phi = {}'.format(coordinates[0],coordinates[1],coordinates[2]))
                    print(' x= {}, y = {}, z =  {}'.format(px,py,pz))
                    print('bunit = {}, change = {}, dr = {} \n \n'.format(Bunit, change, dr))
 
            if i % 1000 == 0 or i==1:
               print("theta = {}".format(coordinates[1]))
               print(B_current, B_notcurrent,r/Rj,'\n')
               
               

        return points
        

    def trace_upper_hemisphere(self, printing='off', starting_cordinates = None):##
        print(' \n Upper Hemisphere:')
        if starting_cordinates == None:
            starting_cordinates = self.starting_cordinates
        coordinates = starting_cordinates
        points = []
        i = 0
        while True:
            i += 1           
            px, py, pz = self.help.sph_to_cart(coordinates[0],coordinates[1],coordinates[2])

            points.append([px,py,pz])
            
            r = coordinates[0]
            if r <= 3 * Rj:
                break


            B_r, B_theta, B_phi = self.field.Internal_Field(r/Rj, coordinates[1], coordinates[2], model=self.model)
            B_current = self.field.CAN_sheet(r/Rj, coordinates[1], coordinates[2])
            B_notcurrent = np.array([B_r, B_theta, B_phi])
            B_overall = np.add(B_current, B_notcurrent)
            B_x, B_y, B_z = self.help.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], coordinates[0], coordinates[1],coordinates[2])
            B = np.array([B_x, B_y, B_z])
            coordinates = [px,py,pz]
            Bunit = self.help.unit_vector_cart(B)
            dr = r * 0.001 #(*Rj) #THIS IS HOW WE UPPDATE THE COORDINATES - IF IT TAKES TOO LONG, THIS NEEDS CHANGING IF IT TAKES TOO LONG OR IS GETTING WEIRD CLOSE TO PLANET
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
            if i % 1000 == 0:
               print("theta = {}".format(coordinates[1]))
               print(B_current, B_notcurrent,r/Rj,'\n')

        return points


    def plotTrace(self):
        lower = np.array(self.trace_lower_hemisphere())
        upper = np.array(self.trace_upper_hemisphere())

        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plottable_lists_upper = np.transpose(upper)
        plottable_lists_lower = np.transpose(lower)

        #turning the axis into Rj
        plottable_lists_lower_rj = plottable_lists_lower/Rj
        plottable_lists_upper_rj = plottable_lists_upper/Rj



        ax.plot(plottable_lists_upper_rj[0], plottable_lists_upper_rj[1], plottable_lists_upper_rj[2],color = 'black', label = 'Field Trace')
        ax.plot(plottable_lists_lower_rj[0], plottable_lists_lower_rj[1], plottable_lists_lower_rj[2], color = 'black', label = 'Field Trace')
        #make the sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color = 'yellow', zorder=100, label = 'Jupiter')
        ax.set_xlim3d(-40, 40)
        ax.set_ylim3d(-40, 40)
        ax.set_zlim3d(-40, 40)
        ax.set_xlabel('$X, R_j$', fontsize=10)
        ax.set_ylabel('$Y, R_J$', fontsize=10)
        plt.title('Magnetic Field Trace using {} model, including current sheet'.format(self.model))
        #plt.legend()
        plt.savefig('images/mag_field_trace_{}_current.png'.format(self.model))
        plt.show()


    def plotMultipleLines(self,r = 20*Rj, num = 8):
        startingPoints = []
        spacing = 2*np.pi/num
        for n in range(num):
            startingPoints.append([r, np.pi/2, n*spacing])
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        colours = ['b','g','r','c','m','k',]
        color_index = 0
        for point in startingPoints:

            lower = np.array(self.trace_lower_hemisphere(starting_cordinates = point))
            upper = np.array(self.trace_upper_hemisphere(starting_cordinates = point))


            
            plottable_lists_upper = np.transpose(upper)
            plottable_lists_lower = np.transpose(lower)

            #turning the axis into Rj
            plottable_lists_lower_rj = plottable_lists_lower/Rj
            plottable_lists_upper_rj = plottable_lists_upper/Rj

            linecolor = colours[color_index]
            color_index +=1
            if color_index > len(colours)-1:
                color_index = 0

            ax.plot(plottable_lists_upper_rj[0], plottable_lists_upper_rj[1], plottable_lists_upper_rj[2],color = linecolor, label = 'Field Trace')
            ax.plot(plottable_lists_lower_rj[0], plottable_lists_lower_rj[1], plottable_lists_lower_rj[2], color = linecolor, label = 'Field Trace')
        #make the sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color = 'yellow', zorder=100, label = 'Jupiter')
        ax.set_xlim3d(-40, 40)
        ax.set_ylim3d(-40, 40)
        ax.set_zlim3d(-40, 40)
        ax.set_xlabel('$X, R_j$', fontsize=10)
        ax.set_ylabel('$Y, R_J$', fontsize=10)
        plt.title('Magnetic Field Trace using {} model, including current sheet'.format(self.model))
        #plt.legend()
        plt.savefig('images/mag_field_multi_trace_{}_inc_current.png'.format(self.model))
        plt.show()
    def plot2d(self):
        '''
        will only see sensible results if y = 0 throughout and for dipole field 
        plots the x-z plane 
        '''

        lower = np.array(self.trace_lower_hemisphere())
        upper = np.array(self.trace_upper_hemisphere())

        
        fig, ax = plt.subplots()
        plottable_lists_upper = np.transpose(upper)
        plottable_lists_lower = np.transpose(lower)

        #turning the axis into Rj
        plottable_lists_lower_rj = plottable_lists_lower/Rj
        plottable_lists_upper_rj = plottable_lists_upper/Rj

        ax.plot(plottable_lists_upper[0], plottable_lists_upper[2],color = 'black', label = 'Field Trace')
        ax.plot(plottable_lists_lower[0], plottable_lists_lower[2], color = 'black')
        #make the circle
        ax.add_patch(Circle((0,0), Rj, color='y', zorder=100, label = "Jupiter"))
        ax.legend()
        plt.savefig('images/individual_mag_field_trace_2d_inc_current_sheet.png')
        plt.show()

 


test = InternalAndCS([30*Rj, np.pi/2, np.pi/2], model = 'VIP4')
test.plotTrace()

