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
class DifferentFieldsTrace:
    ''' 

    TO DO - CHECK IF IT WORKS, PLOT THE TRACE, EXTEND TO MORE COMPLICATED MODELS? COULD DO THE 463 MODEL WHICH WILL INCLUDE A BPHI COMPONENT DUE TO SPINNING PLANET 

    '''

    def __init__(self, starting_cordinates, coord_type = "sph", equatorial_strength = 4.17e-7, model = 'VIP4'):
        self.help = HelpfulFunctions()
        models = ['JRM09', 'VIP4', 'VIT4', 'Ulysses 17ev', 'V1-17ev','O6','O4', 'SHA', 'dipole']
        if model not in models:
            print('model not recognised')
            sys.exit()
        else:
            self.model = model
        self.field = field_models()

        if not (coord_type == 'sph' or coord_type == 'cart'):
            print('please input co-ordinates in spherical polar or cartesian ("sph" / "cart") \n I will extend to further co-ordinate types later')
        if coord_type == 'cart':
            x,y,z = starting_cordinates[0], starting_cordinates[1], starting_cordinates[2]
            r,theta,phi = self.help.cart_to_sph(x, y, z)
            self.starting_cordinates = [r,theta,phi]
        if coord_type == "sph":
            self.starting_cordinates = starting_cordinates

    def trace_lower_hemisphere(self, printing = 'off', starting_cordinates = None):
        print('\n Lower Hemisphere')
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


            B_r, B_theta, B_phi = self.field.Internal_Field(r, coordinates[1], coordinates[2], model=self.model)
            B_current = self.field.CAN_sheet(r, coordinates[1], coordinates[2])
            B_notcurrent = np.array([B_r, B_theta, B_phi])
            B_overall = np.add(B_current, B_notcurrent)
            B_x, B_y, B_z = self.help.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], coordinates[0], coordinates[1],coordinates[2])    
            B_x, B_y, B_z = self.help.Bsph_to_Bcart(B_r, B_theta, B_phi, coordinates[0], coordinates[1],coordinates[2])# quad =1)
            '''
            if not coordinates[2]  == 0:
                print('PHI NOT EQUAL 0')
                sys.exit()
            '''
            B = np.array([B_x, B_y, B_z])
            coordinates = [px,py,pz]
            Bunit = self.help.unit_vector_cart(B)
            dr = r * 0.001 #(*Rj) #THIS IS HOW WE UPPDATE THE COORDINATES - IF IT TAKES TOO LONG, THIS NEEDS CHANGING IF IT TAKES TOO LONG OR IS GETTING WEIRD CLOSE TO PLANET
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


            B_r, B_theta, B_phi = self.field.Internal_Field(r, coordinates[1], coordinates[2], model=self.model)
            B_current = self.field.CAN_sheet(r, coordinates[1], coordinates[2])
            B_current_rh_R, B_current_rh_theta, B_current_rh_phi = self.help.B_S3LH_to_S3RH(B_current[0], B_current[1], B_current[2], hemipshere = 'upper')
            B_current_RH = [B_current_rh_R, B_current_rh_theta, B_current_rh_phi]
            
            B_notcurrent = np.array([B_r, B_theta, B_phi])
            B_overall = np.add(B_current_RH, B_notcurrent)
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
               print("theta = {}, r = {}Rj".format(coordinates[1], coordinates[0]/Rj))

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

 


test = DifferentFieldsTrace([30*Rj, np.pi/2, np.pi/2], model = 'VIT4')
test.plotTrace()

