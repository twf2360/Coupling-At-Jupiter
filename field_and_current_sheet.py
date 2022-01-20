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

    
    def trace_magnetic_field(self, printing = 'off', starting_cordinates = None, one_way = 'off', break_point = 3):
        if starting_cordinates == None:
            starting_cordinates = self.starting_cordinates
        coordinates = starting_cordinates
        coordinates[2] = 2*np.pi - coordinates[2] #changing from LH input to RH
        points = [] #this is the list that will eventually be plotted
        Br_list = []
        B_list = []
        i = 0
        direction = 1
        while True:
            ''' 
            loops for as long as r is larger than a defined value
            '''
            r = coordinates[0]
            if r <= break_point * Rj: #defines when the loop is broken out of 
                if one_way == 'on':
                    break
                if direction == 1:
                    direction = -1
                    coordinates = starting_cordinates
                    points.reverse()
                    Br_list.reverse()
                else:    
                    Br_list.reverse()
                    break

            
            i += 1 #this i is just used to define when to print things! 
            
            px, py, pz = self.help.sph_to_cart(coordinates[0],coordinates[1],coordinates[2])
            points.append([px,py,pz]) #save all of the points so we can plot them later!
            
            
            B_r, B_theta, B_phi = self.field.Internal_Field(r/Rj, coordinates[1], coordinates[2], model=self.model) #calculates the magnetic field due to the internal field in spherical polar that point)
            B_current = self.field.CAN_sheet(r/Rj, coordinates[1], coordinates[2]) #calculates the magnetic field due to the current sheet in spherical polar
            B_notcurrent = np.array([B_r, B_theta, B_phi]) 
            B_overall = np.add(B_current, B_notcurrent) #adds up the total magnetic field 
            Br_list.append(B_overall[0])
            B_x, B_y, B_z = self.help.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], coordinates[0], coordinates[1],coordinates[2]) #converts magnetic field to cartesian
            B = np.array([B_x, B_y, B_z])
            B_list.append(B)
            #print(B)
            coordinates = [px,py,pz] #change the definition of the coordinates from spherical to cartesian 
            Bunit = self.help.unit_vector_cart(B) #calculates the unit vector in cartesian direction
            dr = r * 0.001  #THIS IS HOW WE UPPDATE THE COORDINATES - IF IT TAKES TOO LONG, THIS NEEDS CHANGING IF IT TAKES TOO LONG OR IS GETTING WEIRD CLOSE TO PLANET
            change = dr * Bunit * direction #the change from this coordinate to the next one is calculated
            coordinates = np.add(coordinates, change) #add the change to the current co ordinate
            pr, ptheta, pphi = self.help.cart_to_sph(coordinates[0], coordinates[1], coordinates[2]) #change the coordinatres back in spherical
            coordinates = [pr,ptheta,pphi] 

            
            if printing == 'on':
                if (i % 1000) == 0 or i == 1:
                    print('B cartesian = {}, B sph = [{} {} {}]'.format(B, B_r, B_theta, B_phi))
                    print('r = {}, theta = {}, phi = {}'.format(coordinates[0],coordinates[1],coordinates[2]))
                    print(' x= {}, y = {}, z =  {}'.format(px,py,pz))
                    print('bunit = {}, change = {}, dr = {} \n \n'.format(Bunit, change, dr))
 
        return points, Br_list, B_list



    def plotTrace(self):
        plot_points = np.array(self.trace_magnetic_field(printing='off')[0])

        plottable_list = np.transpose(plot_points)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        

        #turning the axis into Rj
        plottable_list_rj = plottable_list/Rj

        ax.plot(plottable_list_rj[0], plottable_list_rj[1], plottable_list_rj[2],color = 'black', label = 'Field Trace')
        
        #make the sphere and setup the plot
        x,y, z = self.make_sphere()
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


    def plotMultipleLines(self,r = 30*Rj, num = 8):
        startingPoints = []
        spacing = 2*np.pi/num
        for n in range(num):
            startingPoints.append([r, np.pi/2, n*spacing])
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        colours = ['b','g','r','c','m','k',]
        color_index = 0
        for point in startingPoints:
            print('Starting point = {}'.format(point))
            plot_points = np.array(self.trace_magnetic_field(starting_cordinates=point)[0])

            
            plottable_list = np.transpose(plot_points) #turns from px,py,pz to [[x0, x1, x2, ...], [y0, y1, y2, ....], [z0, z1, z2, ....]]


            #turning the axis into Rj
            plottable_list_rj = plottable_list/Rj

            linecolor = colours[color_index]
            color_index +=1
            if color_index > len(colours)-1:
                color_index = 0

            ax.plot(plottable_list_rj[0], plottable_list_rj[1], plottable_list_rj[2],color = linecolor, label = 'Field Trace')
            
        
        #make the sphere
        x,y,z = self.make_sphere()

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
        points = np.array(self.trace_magnetic_field())
        

        fig, ax = plt.subplots()
        plottable_list = np.transpose(points)

        #turning the axis into Rj
        plottable_list_rj = plottable_list/Rj

        ax.plot(plottable_list_rj[0], plottable_lists_rj[2],color = 'black', label = 'Field Trace')
       
        #make the circle
        ax.add_patch(Circle((0,0), Rj, color='y', zorder=100, label = "Jupiter"))
        ax.legend()
        plt.savefig('images/individual_mag_field_trace_2d_inc_current_sheet.png')
        plt.show()

    def make_sphere(self):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        return x, y, z

    def find_mag_equator(self, point):

        print(' \n Starting point = {}'.format(point))
        points, Br_list = self.trace_magnetic_field(starting_cordinates=point)
        index = self.find_index_negative(listInput = Br_list)
        def interpolate(i):
            point1 = points[i-1]
            point2 = points[i]
            theta1 = self.help.cart_to_sph(point1[0], point1[1], point1[2])[1]
            theta2 = self.help.cart_to_sph(point2[0], point2[1], point2[2])[1]
            Br_1 = Br_list[i-1]
            Br_2 = Br_list[i]
            theta = (theta1 + (abs(Br_2/Br_1) *theta2))/(1 + abs(Br_2/Br_1))
            #print(theta, theta1, theta2, Br_1, Br_2)
            return theta 
        theta = interpolate(index)
        print('mag field lies in plane theta = {}'.format(theta))
        return theta
            
    def traceFieldEquator(self):
        tracer = self.trace_magnetic_field(printing='off')
        plot_points, Br_list = np.array(tracer[0]), np.array(tracer[1])
        plottable_list = np.transpose(plot_points)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        

        #turning the axis into Rj
        plottable_list_rj = plottable_list/Rj

        ax.plot(plottable_list_rj[0], plottable_list_rj[1], plottable_list_rj[2],color = 'black', label = 'Field Trace')
        
        #make the sphere and setup the plot
        x,y, z = self.make_sphere()
        ax.plot_surface(x, y, z, color = 'yellow', zorder=100, label = 'Jupiter')
        ax.set_xlim3d(-40, 40)
        ax.set_ylim3d(-40, 40)
        ax.set_zlim3d(-40, 40)
        ax.set_xlabel('$X, R_j$', fontsize=10)
        ax.set_ylabel('$Y, R_J$', fontsize=10)
        plt.title('Magnetic Field Trace using {} model, including current sheet'.format(self.model))
        
        index = self.find_index_negative(listInput = Br_list)
        def interpolate(i):
            point1 = plot_points[i-1]
            point2 = plot_points[i]
            theta1 = self.help.cart_to_sph(point1[0], point1[1], point1[2])[1]
            theta2 = self.help.cart_to_sph(point2[0], point2[1], point2[2])[1]
            Br_1 = Br_list[i-1]
            Br_2 = Br_list[i]
            theta = (theta1 + (abs(Br_2/Br_1) *theta2))/(1 + abs(Br_2/Br_1))
            #print(theta, theta1, theta2, Br_1, Br_2)
            return theta 
        theta = interpolate(index)
        r = self.starting_cordinates[0]
        R_rj = r/Rj
        start_phi = self.starting_cordinates[2]
        output = self.help.sph_to_cart(R_rj, theta, start_phi)

        equator_plot = np.array([[-output[0], -output[1], -output[2]],[output[0], output[1], output[2] ]])
        transposed_equator = np.transpose(equator_plot)
        ax.plot(transposed_equator[0], transposed_equator[1], transposed_equator[2], color = 'c', label = 'mag field equator')#, linewidth = 5.0)
        print(transposed_equator)
        #plt.legend()
        plt.savefig('images/mag_field_trace_showing_B_equator.png'.format(self.model))
        plt.show()
            
    def find_index_negative(self, listInput):
        for i in range(len(listInput)):
            
            if listInput[i] <= 0:
                #print(i, listInput[i], listInput[i-1])
                return i
        return None


'''    
test = InternalAndCS([30*Rj, np.pi/2, 212* np.pi/180], model = 'VIP4')
#test.find_mag_equator(point=[30*Rj, np.pi/2, 112* np.pi/180])
#test.plotTrace()
#test.plotMultipleLines()
test.traceFieldEquator()
'''