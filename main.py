from cProfile import label
from pyexpat import model
from scipy.signal import savgol_filter
import math
from turtle import color
from unittest import result
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import json
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, PathPatch
from os.path import exists
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib as mpl
from sympy import false
#from sqlalchemy import false
from Lorch_mag_field_models import field_models
import scipy
import matplotlib.colors as mcolors
import scipy.special
from matplotlib import ticker, cm
from copy import deepcopy
#from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               #AutoMinorLocator)
from labellines import labelLine, labelLines
from Ray_SimpleAlfvenModel.Grid import Grid as RayGrid
from Ray_SimpleAlfvenModel.Field import Field as RayField
from Ray_SimpleAlfvenModel.Planet import Planet as RayPlanet
import Ray_SimpleAlfvenModel.BFlux as RayBF

plt.rcParams['legend.fontsize'] = 14
personal_cmap = ['deeppink', 'magenta', 'darkmagenta' ,'darkorchid', 'indigo','midnightblue', 'darkblue', 'slateblue', 'dodgerblue', 'deepskyblue',  'aqua', 'aquamarine' ]
Rj = 7.14 * (10 ** 7)
omega_J = 2*np.pi/(9.925*60*60)
mu_0 = 1.25663706212 * 10 ** -6
B0 = 417000 #in nT

plt.style.use('ggplot')

plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'axes.titlesize':14, 'axes.spines.top': True,'axes.spines.bottom' : True, 'axes.spines.left': True, 'axes.spines.right': True, 'axes.linewidth':1, 'axes.edgecolor':'black', 'xtick.top':True, 
'ytick.right':True, 'xtick.minor.visible':True, 'ytick.minor.visible':True, 'xtick.major.size':10, 'ytick.major.size':10, 'xtick.minor.size':5, 'ytick.minor.size':5,
'xtick.major.width': 2, 'ytick.major.width': 2})
plt.rcParams.update({ 'xtick.direction':'in','ytick.direction':'in'})

plt.rcParams.update({'font.size': 20})
### Author @twf2360
class main:
    def __init__(self, model = 'dipole', aligned='yes', avgIonMass = 21):
        ''' 
        Class For Masters Thesis titled = 

        Inputs:
        model = "dipole", "VIP4". "dipole" is a spin aligned dipole. VIP4 calculated according to Chris Lorch's mag_field_models.py
        aligned = "yes", "no". If dipole is selected, aligned must be yes. Are the spin and centrifigual aligned, if no centrifugal equator
                    calculated

        avgIonMass = average mass of ions in amu
        '''
        self.model = model

        if self.model == 'dipole':
            self.aligned = 'yes'
            self.plot_label = 'Spin Aligned Dipole Magnetic Field Approximation'

        else:
            self.aligned = aligned
            self.field = field_models()
            if aligned == 'yes':
                self.plot_label = 'VIP4 Magnetic Field With Spin Aligned Centrifugal Equator Approximation'
            else:
                self.plot_label = 'VIP4 Magnetic Field With Non-Aligned Spin and Centrifugal Equators'
        self.avgIonMass = avgIonMass * 1.67 * 10**-27
        self.CML = 0
        

    
    """ 
    the below are helpful functions such as co-ordinate transforms 
    """
    def makegrid_3d(self,NumPoints, gridsize):
        ''' 
        makes a 3d grid using np.meshgrid, returns x,y,z arrays 
        grid ranges from -gridsize/2 to +gridsize/2, with the number of points in each array defined by numpoints
        ''' 
        points_min = -gridsize/2
        points_max = gridsize/2
        x,y,z = np.meshgrid(np.linspace(points_min, points_max, NumPoints),
                            np.linspace(points_min, points_max, NumPoints),
                            np.linspace(points_min, points_max, NumPoints))
        return x,y,z
    
    def makegrid_2d(self, numpoints, gridsize):
        '''
        makes a 2d grid using np.meshgrid, returns x,y arrays 
        grid ranges from 0 to gridsize, with the number of points in each array defined by numpoints
        '''
        x,y = np.meshgrid(np.linspace(0, gridsize, numpoints),
                            np.linspace(0, gridsize, numpoints))
        return x,y

    def makegrid_2d_negatives(self, numpoints, gridsize):
        '''
        makes a 2d grid using np.meshgrid, returns x,y arrays 
        grid ranges from -gridsize/2 to +gridsize/2, with the number of points in each array defined by numpoints
        '''
        x,y = np.meshgrid(np.linspace(-gridsize, gridsize, numpoints *2),
                            np.linspace(-gridsize, gridsize, numpoints*2))
        return x,y
        
    def sph_to_cart(self, r, theta, phi):
        ''' 
        Input spherical co-ordinates (r, theta, phi)
        Note: theta is COLATITUDE, phi is measured away from X (right hand)
        Returns cartesian co-ordinates x,y,z
        
        ''' 
        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)
        return x,y,z



    def cart_to_sph(self,x ,y,z):
        ''' 
        Input Cartesian co-ordinates (x, y, z)
        Returns spherical co-ordinates r, theta phi
        Note: Note: theta is COLATITUDE, phi is measured away from X (right hand)
        ''' 
        r = np.sqrt(x**2+y**2+z**2)
        theta = np.arctan2(np.sqrt(x**2+y**2),z)
        phi = np.arctan2(y,x)
        return r, theta, phi

    def Bsph_to_Bcart(self, Br, Btheta, Bphi, r, theta,phi):
        ''' 
        Input Magnetic field vector in spherical co-ordinates (Br, Btheta, Bphi), and spherical co-ordinate points (r, theta, phi)
        Where theta is COLATITUDE and phi is measured away from X (right hand)
        Returns the magnetic field vector components in cartesian co-ordinates Bx, By, Bz
        '''
        theta = float(theta)
        phi = float(phi)
        Bx = Br * np.sin(theta) * np.cos(phi) + Btheta * np.cos(theta) * np.cos(phi) - Bphi *np.sin(phi)
        By = Br * np.sin(theta) * np.sin(phi) + Btheta * np.cos(theta) * np.sin(phi) + Bphi *np.cos(phi)
        Bz = Br * np.cos(theta)  - Btheta * np.sin(theta)
        return Bx, By, Bz



    def unit_vector_cart(self, vector):
        ''' 
        input a vector in cartesian coordinates
        returns the unit vector
        '''  
        norm = np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
        return vector/norm

    def centrifugal_equator(self, r, phi):
        ''' 
        Input the r, phi coordinates of a point on the spin equator of Jupiter
        Note: R is in Rj, Phi is right handed 
        Returns theta, the latitidue of the centrifugal equator at that distance r from the planet
        equation is eq 2 from https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2020JA028713 eq 
        
        '''
        #phiLH = 2 * np.pi - phi
        phiRH = phi
        a = 1.66 * np.pi / 180
        b = 0.131
        c = 1.62
        d = 7.76 * np.pi /180
        e = 249 * np.pi/180
        centrifualEq = (a * np.tanh(b*r -c) + d) * np.sin(phiRH - e)
        

        return centrifualEq 

    def change_equators(self, r, theta, phi):
        ''' 
        input r, theta, phi (r in Rj, theta colatitude, phi is right handed) with respect to the spin axis
        retuns r, theta phi with respect to the centrifugal axis
        '''
        r_cent = r 
        phi_cent = phi
        theta_shift = self.centrifugal_equator(r, phi)
        theta_cent = theta + theta_shift
        return r, theta_cent, phi_cent
        
    def change_equators_cart_output(self, r, theta, phi):
        '''
        Input r (Rj), theta (colatitude) and phi (rh) with respect to spin axis 
        returns x,y,z with respect to the centrifugal axis
        '''
        r_cent = r 
        phi_cent = phi
        theta_shift = self.centrifugal_equator(r, phi)
        theta_cent = theta + theta_shift
        #if theta_shift < 0:
         #   print('hello there')
        #print('theta shift = {} \n theta cent = {}'.format(theta_shift, theta_cent))
        
        scaleheight = self.scaleheight(r_cent)
        x_cent, y_cent, z_cent = self.sph_to_cart(r_cent, theta_cent, phi_cent)
        return x_cent, y_cent, z_cent


    def complex_mag_equator(self, r, phi_lh):
        ''' 
        input r(rj), theta (colatitude), phi (lh)
        returns latitude of the magnetic equator with respect to the spin equator
        ''' 
        phi =  2 *np.pi - phi_lh #change phi to RH. 
        #phi = phi_lh
        guesses_degress = np.array([30,40,50,60,70,80,90,100,110,120,130, 140, 150], dtype = float)
        guesses_radians_1 = guesses_degress * np.pi/180
        oneDegreeInRadians = 1*np.pi/180
        def find_swap(angles):
            b_r_list = []
            for i in range(len(angles)):
                B_overall = self.mag_field_at_point(r, angles[i], phi)

                b_r_list.append(B_overall[0])
            crossing = np.where(np.diff(np.sign(b_r_list)))[0]
            b_r_list = []      
            return angles[crossing], angles[crossing+1]
        stop1, stop2 = find_swap(guesses_radians_1)
        guesses_2 = np.arange(stop1, stop2 + oneDegreeInRadians, oneDegreeInRadians)
        stop3, stop4 = find_swap(guesses_2)
        guesses_3 = np.arange(stop3, stop4+ 0.1* oneDegreeInRadians, 0.1*oneDegreeInRadians )
        answer_low, answer_high = find_swap(guesses_3)
        answer = (answer_high + answer_low) /2
        return answer[0]
    

    def calc_furthest_r(self, points):
        ''' 
        input a list of points [[x,y,z],[x,y,z]...]
        returns the furthest distance r reached by those points
        ''' 
        rs = []
        for point in points:
            x = point[0]
            y = point[1]
            z = point[2]
            rs.append(np.sqrt(x**2 + y**2 + z**2))
        largest_r = max(rs)
        largest_r_rj = largest_r/Rj
        return largest_r_rj

    
    
    def mag_field_at_point(self, r, theta, phi):
        ''' 
        input r, theta phi where r is in Rj, theta is colatitude, and phi is RIGHT HANDED
        returns magnetic field vector (in nT) at that point  
        '''
        if self.model == 'dipole':
            #Overall strength of the vector will scale with distance 
            ScaleFactor = B0 * (1/r)**3

            #magnetic field in radial and polar direction
            B_r = - 2* ScaleFactor * np.cos(theta) #there was 2 here
            B_theta = - ScaleFactor * np.sin(theta)
            B_phi = 0
            return [B_r, B_theta, B_phi]
        if self.model == 'VIP4':
            B_r, B_theta, B_phi = self.field.Internal_Field(r, theta, phi, model=self.model) #calculates the magnetic field due to the internal field in spherical polar that point)
            B_current = self.field.CAN_sheet(r, theta, phi) #calculates the magnetic field due to the current sheet in spherical polar
            B_notcurrent = np.array([B_r, B_theta, B_phi]) 
            B_overall = np.add(B_current, B_notcurrent)
            return B_overall


  
    def plot_Bvs_r_cent_equator(self, phi_lh_deg):
        '''
        input longitude along which you wish to investigate (in lh degrees)
        plot B vs R along the centrifugal equator
        '''
        rs = np.arange(6,60,0.5)
        phi_lh_rad = phi_lh_deg *np.pi/180
        phi_rh_rad = 2*np.pi - phi_lh_rad
        Bs = []
        for r in rs:
            latitude_cent_equator = self.centrifugal_equator(r, phi_lh_rad)
            colat = np.pi/2 - latitude_cent_equator
            B = self.mag_field_at_point(r,colat, phi_rh_rad)
            magB = B[0]
            Bs.append(magB)
        ''' plot stuff goes here ''' 
        fig, ax = plt.subplots()
        ax.plot(rs,Bs)
        plt.show()

    def trace_magnetic_field(self, printing = 'off', starting_cordinates = None, one_way = 'off', break_point = 3, step = 0.001, pathing = 'forward', footprint = False):
        ''' 
        Calculate the magnetic field trace from startpoint (r,theta, phi) with r in ms, theta is colatitude and phi is LH
        INPUT PHI IS LH
         ''' 

        
        coordinates = starting_cordinates
        #print('start', coordinates[2])
        coordinates[2] = 2*np.pi - coordinates[2] #changing from LH input to RH
        #print('after', coordinates[2])
        points = [] #this is the list that will eventually be plotted
        Br_list = []
        B_list = []
        dr_list = []
        r_list = []
        i = 0
        direction = 1
        if pathing == 'backward':
            path_direction = -1
        else:
            path_direction = 1
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
            
            px, py, pz = self.sph_to_cart(coordinates[0],coordinates[1],coordinates[2])
            points.append([px,py,pz]) #save all of the points so we can plot them later!
            
            B_overall = self.mag_field_at_point(r/Rj, coordinates[1], coordinates[2]) 
            Br_list.append(B_overall[0])
            B_x, B_y, B_z = self.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], coordinates[0], coordinates[1],coordinates[2]) #converts magnetic field to cartesian
            B = np.array([B_x, B_y, B_z])
            B_list.append(B)
            #print(B)
            coordinates = [px,py,pz] #change the definition of the coordinates from spherical to cartesian 
            Bunit = self.unit_vector_cart(B) #calculates the unit vector in cartesian direction
            dr = r * step  #THIS IS HOW WE UPPDATE THE COORDINATES - IF IT TAKES TOO LONG, THIS NEEDS CHANGING IF IT TAKES TOO LONG OR IS GETTING WEIRD CLOSE TO PLANET
            dr_list.append(dr)
            r_list.append(r)
            change = dr * Bunit * direction * path_direction#the change from this coordinate to the next one is calculated
            coordinates = np.add(coordinates, change) #add the change to the current co ordinate
            pr, ptheta, pphi = self.cart_to_sph(coordinates[0], coordinates[1], coordinates[2]) #change the coordinatres back in spherical
            coordinates = [pr,ptheta,pphi] 
            #print('r, theta, phi = ',coordinates[0]/Rj, coordinates[1] * 180/np.pi, coordinates[2])
            '''
            i am keeping the next three lines in memorandum the absolute choas they caused. 
            f = open('points.txt', 'a')
            f.write('r, colat, phi lh = {}, {}, {} \n'.format(coordinates[0]/Rj, coordinates[1] * 180/np.pi, coordinates[2]))
            f.close()
            '''
            if printing == 'on':
                if (i % 1000) == 0 or i == 1:
                    print('B cartesian = {}, B sph = [{} {} {}]'.format(B, B_r, B_theta, B_phi))
                    print('r = {}, theta = {}, phi = {}'.format(coordinates[0],coordinates[1],coordinates[2]))
                    print(' x= {}, y = {}, z =  {}'.format(px,py,pz))
                    print('bunit = {}, change = {}, dr = {} \n \n'.format(Bunit, change, dr))
 
        if footprint:
            return points[-1]
        return points, Br_list, B_list, dr_list, r_list



    def plotTrace(self, startpoint):
        '''
        plots the magnetic field trace from startpoint given in r(m), theta (colatitude), phi (lh)
        '''
        plot_points = np.array(self.trace_magnetic_field(printing='off', starting_cordinates = startpoint)[0])

        plottable_list = np.transpose(plot_points)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        

        #turning the axis into Rj
        plottable_list_rj = plottable_list/Rj

        ax.plot(plottable_list_rj[0], plottable_list_rj[1], plottable_list_rj[2],color = 'black', label = 'Field Trace')
        
        #make the sphere and setup the plot
        x,y, z = self.make_sphere()
        ax.plot_surface(x, y, z, color = 'yellow', zorder=100, label = 'Jupiter')
        ax.set_xlim3d(-20, 20)
        ax.set_ylim3d(-20, 20)
        ax.set_zlim3d(-20, 20)
        ax.set_xlabel('\n \n$X (R_J)$')#, fontsize=10)
        ax.set_ylabel('\n \n$Y (R_J)$')#, fontsize=10)
        #plt.title('Magnetic Field Trace using {} model, including current sheet'.format(self.model))
        #plt.legend()
        ####plt.savefig('images/mag_field_trace_{}_current.png'.format(self.model))
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
        ###plt.savefig('images/mag_field_multi_trace_{}_inc_current.png'.format(self.model))
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
        ###plt.savefig('images/individual_mag_field_trace_2d_inc_current_sheet.png')
        plt.show()

    def make_sphere(self):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        return x, y, z

    def find_mag_equator(self, point):
        ''' input phi is lh '''

        print(' \n Starting point = {}'.format(point))
        calc =  self.trace_magnetic_field(starting_cordinates=point)
        points, Br_list = calc[0], calc[1]
        index = self.find_index_negative(listInput = Br_list)
        def interpolate(i):
            point1 = points[i-1]
            point2 = points[i]
            theta1 = self.cart_to_sph(point1[0], point1[1], point1[2])[1]
            theta2 = self.cart_to_sph(point2[0], point2[1], point2[2])[1]
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
            theta1 = self.cart_to_sph(point1[0], point1[1], point1[2])[1]
            theta2 = self.cart_to_sph(point2[0], point2[1], point2[2])[1]
            Br_1 = Br_list[i-1]
            Br_2 = Br_list[i]
            theta = (theta1 + (abs(Br_2/Br_1) *theta2))/(1 + abs(Br_2/Br_1))
            #print(theta, theta1, theta2, Br_1, Br_2)
            return theta 
        theta = interpolate(index)
        print(theta*180/np.pi)
        #r = self.starting_cordinates[0]
        R_rj = 30
        start_phi = self.starting_cordinates[2]
        output = self.sph_to_cart(R_rj, theta, start_phi)

        equator_plot = np.array([[-output[0], -output[1], -output[2]],[output[0], output[1], output[2] ]])
        transposed_equator = np.transpose(equator_plot)
        ax.plot(transposed_equator[0], transposed_equator[1], transposed_equator[2], color = 'c', label = 'mag field equator')#, linewidth = 5.0)
        #print(transposed_equator)
        #plt.legend()
        ###plt.savefig('images/mag_field_trace_showing_B_equator.png'.format(self.model))
        plt.show()
            
    def find_index_negative(self, listInput):
        for i in range(len(listInput)):
            
            if listInput[i] <= 0:
                #print(i, listInput[i], listInput[i-1])
                return i
        return None
    
    
    def find_furthest_r_single_input(self, startpoint):
        ''' 
        input start point (r, theta, phi) where r is in rj and phi is left handed 
        return furthest rdial distance reached by that field line 
        '''
        startpoint[0] = startpoint[0]*Rj
        plot_results = self.trace_magnetic_field(starting_cordinates=startpoint, one_way='off', break_point=2, step = 0.001)
        points = np.array(plot_results[0])
        furthest_r = self.calc_furthest_r(points)
        return furthest_r

    def radial_profile_b_eq_plane(self, start = 2, stop = 70, numpoints = 200, phi = 200.8):
        rs = np.linspace(start, stop, numpoints)
        phi_rh_rad = 2*np.pi - (phi * np.pi/180)
        theta = np.pi/2
        magBs = []
        for r in rs:
            B_overall = self.mag_field_at_point(r, theta, phi_rh_rad)
            B_x, B_y, B_z = self.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi_rh_rad)
            B = np.array([B_x, B_y, B_z])
            magB = np.linalg.norm(B)
            magBs.append(magB)
        fig, ax = plt.subplots(figsize = (8,5))
        ax.plot(rs, magBs, label = 'Magnitude of Magnetic Field $(nT)$')
        ax.legend()
        ax.set(xlabel='Radius $(R_J)$', ylabel='Magnitude of Magnetic Field $(nT)$', title='magnetic Field Vs Radial Distsance in Equatorial plane')
        ax.yaxis.set_ticks_position('both')
        plt.yscale("log")
        ax.grid()
        ###plt.savefig('images-24-jan-update/mag_density_profile.png')
        plt.show()

    def radial_profile_B_n(self, phi, start = 6, stop = 60, numpoints = 100):
        rs = np.linspace(start, stop, numpoints)
        phi_rh_rad = 2*np.pi - (phi * np.pi/180)
        phi_lh_rad = phi * np.pi/180
        theta = np.pi/2
        magBs = []
        ns= []
        for r in rs:
            if self.aligned  == 'no':
                latitude_cent_equator = self.centrifugal_equator(r, phi_rh_rad)
                colat = np.pi/2 - latitude_cent_equator
            else:
                colat = np.pi/2
            B_overall = self.mag_field_at_point(r,colat, phi_rh_rad)
            B_x, B_y, B_z = self.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, colat, phi_rh_rad)
            B = np.array([B_x, B_y, B_z])
            magB = np.linalg.norm(B)
            magBs.append(magB)
            n = self.density_combined(r, colat, phi_lh_rad)
            ns.append(n/1e6)
            
        fig, ax1 = plt.subplots(figsize = (8,5))
        ax1.plot(rs, magBs, label = 'Magnitude of Magnetic Field $(nT)$')
        ax1.plot([],[], color = 'k', label = 'Density ($cm^{-3}$)')
        plt.legend()
        ax1.set(xlabel='Radial Distance $(R_J)$', ylabel='Magnitude of Magnetic Field $(nT)$', title='Magnetic Field  and Density Vs Radial Distsance in Centrifugal Plane \n {}'.format(self.plot_label))
        ax1.yaxis.set_ticks_position('both')
        plt.yscale("log")
        ax2 = ax1.twinx()
        ax2.plot(rs, ns, label = 'Density ($cm^{-3}$)', color = 'k')
        ax2.set_ylabel('Density ($cm^{-3}$)')
        
        plt.yscale("log")
        plt.grid()
        ###plt.savefig('images-24-jan-update/mag_density_profile.png')
        plt.show()
        return rs, magBs, ns
   
   
    
    def radial_flow_velocity(self, r, mdot):
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

        v =  mdot/(n*self.avgIonMass*A)#/1.67e-27)
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
        density_cm = np.array(densities)/1e6
        density_eq = []
        for r in radii:
            density_eq.append(3.2e8 * r**(-6.9) + 9.9*r**(-1.28))
        if show == 'on':
            fig, ax = plt.subplots(figsize = (8,5))
            ax.plot(radii, density_cm, label = 'Density $(cm^{-3})$', color = 'k')
            ax.plot(radii, density_eq, linestyle = 'dashdot', color = 'm', label = '$3.2x10^8 r^{-6.9} + 9.9r^{-1.28}$')
            ax.legend()
            ax.set(xlabel='Radial Distance $(R_J)$', ylabel='Density ($cm^3$)')#, title='Density Vs Radial Distsance')
            ax.yaxis.set_ticks_position('both')
            plt.yscale("log")
            ###plt.savefig('images-24-jan-update/radial_density_profile.png')
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
        
        fig, ax1, = plt.subplots()
        ax1.semilogy(radii1, densities1, label = 'r > $20R_J$') #change to ax.plot to remove log scale
        ax1.semilogy(radii2, densities2, label = 'r > $50R_J$', color = 'k') #change to ax.plot ^^
        ax1.legend()
        ax1.set(ylabel='Density ($m^3$)' ,title='Density Vs Radial Distance')
        #ax1.yaxis.set_ticks_position('both')
        ax2 = ax1.twinx()
        ax2.legend()
        ax2.set(xlabel='Radius $(R_J)$', ylabel='Density ($m^{-3}$)') #, title='Density Vs Radial Distance')
        #ax2.yaxis.set_ticks_position('both')
        ###plt.savefig('images-24-jan-update/radial_density_profile_two_points.png')
        plt.show()
        

        
    def local_Alfven_vel_simple(self, r, theta = np.pi/2, phi = 0):
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
                    Va = self.local_Alfven_vel_simple(r)
                    alfven_vel_values.append(Va)
                    
                v = self.flow_velocity(r, mdot)
                flow_for_given_mdot.append(v)
            flow_values[mdot] = flow_for_given_mdot
            i = 1
        

        np.save("Old/data/radial_flow/local_alfven.npy", alfven_vel_values, allow_pickle=True)
        np.save("Old/data/radial_flow/flow_velocity.npy", flow_values, allow_pickle=True)
        myjson = json.dumps(flow_values)
        f = open("Old/data/radial_flow/flow_values_dict.json", "w")
        f.write(myjson)
        f.close()
        np.save("Old/data/radial_flow/r_values.npy", r_values, allow_pickle=True)
    

    def plotOutflow(self):
        '''
        requires there to be data already - plots outflow velocity against radial distance
        '''
        va_values = np.load("Old/data/radial_flow/local_alfven.npy", allow_pickle=True)
        r_values = np.load("Old/data/radial_flow/r_values.npy", allow_pickle=True)
        Rj_values = r_values/Rj
        #v_values = np.load("data/radial_flow/flow_velocity.npy", allow_pickle=True)
        with open("Old/data/radial_flow/flow_values_dict.json", "r") as json_dict:
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
        ###plt.savefig('images-24-jan-update/radial_flow_plot.png')
        plt.show()

    ''' more density functions '''

    def scaleheight(self, R):
        a1 = -0.116
        a2 = 2.14
        a3 = -2.05
        a4 = 0.491
        a5 = 0.126
        r = np.log10(R/6) 
        h = a1 + a2 * r + a3 * r**2 + a4 * r**3 + a5 * r**5
        H = 10**h # Don't worry about the runtime warning - H get's really big for Large R values :D 
        return H

    def density_combined(self,r, theta, phi_lh): 
        
        def density(n_0, z, H):
            n = n_0 * np.exp(-z/H)**2
            return n
        ''' phi lh '''
        def density_sep_equators(r, theta, phi):
            ''' 
            Input r, theta (colatitude), phi (lh)
            returns mass density, taking into account the difference between spin and centrifugal equators. 
            '''
            r_cent = r 
            phi_cent = phi
            phi_rh = 2*np.pi - phi
            theta_shift = self.centrifugal_equator(r, phi_rh)
            theta_cent = theta + theta_shift
            scaleheight = self.scaleheight(r_cent)
            n_0 = self.radial_density(r_cent)
            z_cent =  r_cent * np.cos(theta_cent)
            n = density(n_0, abs(z_cent), scaleheight)
            return n 

        def density_same_equators(r, theta):
            '''
            returns the density if the centrifugal equator lies on the spin equator Y

            '''

            n_0 = self.radial_density(r)
            z = abs(r*np.cos(theta))
            scaleheight = self.scaleheight(r)
            den = density(n_0, z, scaleheight)
            return den
        if self.aligned == 'yes':
            n = density_same_equators(r = r, theta = theta)
        else: 
            n = density_sep_equators(r = r, theta = theta, phi = phi_lh)
        return n


    def density_within_6(self, r, theta, phi_lh, n_at_6):
        ''' r in rj '''
        n = n_at_6 * np.exp(r - 6) 
        return n
        
    


    

    def plotting_density(self, density = 'on',scale_height = 'off', start = 6, stop = 60, numpoints = 200):
        radii, n_0s = self.plotRadialDensity(start=start, end = stop, numpoints=numpoints)
        zs =  np.linspace(start, stop, numpoints)
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
            
            density_0_cm = np.array(n_0s)/ (10**6)
            density_cm = np.array(ns)/(10**6)
            fig, (ax1, ax2) = plt.subplots(1,2, figsize =(25,13))
            
            cont = ax2.contourf(radii, zs, density_cm, cmap = 'bone', locator=ticker.LogLocator())

            ax2.set(xlabel = 'Radial Distance($R_J$)', ylabel = 'Height($R_J$)', title = 'Contour plot of density depending on radial density and height')
            ax2.yaxis.set_ticks_position('both')
            plt.colorbar(cont, label = 'Density ($cm^{-3}$)')
            
            ax1.plot(radii, density_0_cm, label = '$n_0$')
            ax1.legend()
            
            ax1.set(xlabel='Radius $(R_J)$', ylabel='Density ($(cm^{-3})$)', title='Density Vs Radial Distsance along Centrifugal Equator')
            ax1.yaxis.set_ticks_position('both')
            ax1.set_yscale("log")
            plt.suptitle('Density Variations as a function of height and radial distance')
            #fig.tight_layout()
            plt.show()
        
        if scale_height == 'on':
            eq_H = []
            for r in radii:
                Hs.append(self.scaleheight(r))
                a1 = -0.116
                a2 = 2.14
                a3 = -2.05
                a4 = 0.491
                a5 = 0.126
                R = np.log10(r/6) 
                h = a1 + a2 * R + a3 * R**2 + a4 * R**3 + a5 * R**5
                eq_H.append(10**h) 

            fig, ax = plt.subplots(figsize =(25,13))
            ax.plot(radii, Hs, label = 'Calculated Scale Height', color = 'k')
            ax.plot(radii, eq_H, label = 'Equation Result' , linestyle = 'dashdot', color = 'm')
            plt.legend()
            #plt.xscale('log')
            ax.set(xlabel='Radial Distance $(R_J)$', ylabel='Scale Height ($R_J$)')#, title='Scale height depenence on radial distance')
            plt.xlim(0, 70)
            #plt.xticks((5,10,20,30,50,100))
            plt.show()
    

    def equators_cent_calculated(self):
        r = 30
        theta = np.pi/2
        phi_LH =  21* np.pi/180
        phi_rh = 2 *np.pi - phi_LH
        Btheta_eq = self.find_mag_equator(point=[r*Rj, theta,phi_LH])
        b_eq = np.array([[-r * np.sin(Btheta_eq), - r * np.cos(Btheta_eq)], [r * np.sin(Btheta_eq),  r * np.cos(Btheta_eq)]])
        spin_eq = np.array([[-r,0],[r,0]])
        b_eq_t = np.transpose(b_eq)
        spin_eq_t = np.transpose(spin_eq)

        numpoints = 100
        centrifugal_equator = []
        cent_points = np.linspace(-r, r, num=numpoints)
        for x_point in cent_points:
            H = self.height_centrifugal_equator(x_point, phi_rh)
            centrifugal_equator.append([x_point, H])
        centrifugal_eq_t = np.transpose(np.array(centrifugal_equator))


        
        fig, ax = plt.subplots()
        
        ax.plot(centrifugal_eq_t[0] , centrifugal_eq_t[1], color = 'c', label = 'Centrifugal Equator')
        ax.plot(spin_eq_t[0], spin_eq_t[1], color = 'm', label = 'Spin Equator')
        ax.plot(b_eq_t[0], b_eq_t[1], color = 'k', label = 'Magnetic Field Equator')

        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        phi_lh_deg = phi_LH * 180/np.pi 
        ax.set(xlim = (-30,30), ylim = (-5,5), xlabel = 'X $R_J$ \n CML is phi = 0', ylabel = 'Y $R_J$',
         title = 'Different Equators at Jupiter, SYSIII longitude =   {:.0f}{} on RHS'.format(phi_lh_deg,  u"\N{DEGREE SIGN}"))
        ax.set_aspect(aspect='equal')
        plt.legend()
        ###plt.savefig('images/equators.png')
        plt.show()

    def equator_comparison_mag_cent(self, phi = 200, num = 200):
        '''
        plot the magnetic equator vs the centrifugual equator against r, along a longitude defined by phi (lh)
        '''
        r_thetas_dict = self.density_contour_meridian(phi_lh = phi, num= num, field_line= 'off', show = 'off')[0]
        r_thetas_dict_positive_only = {k: v for (k, v) in r_thetas_dict.items() if k >= 2} #this should be 6
        #print(r_thetas_dict_positive_only)
        rs = list(r_thetas_dict_positive_only.keys())
        thetas_c_m = list(r_thetas_dict_positive_only.values())
        divided_thetas = []
        for i in thetas_c_m:
            divided_thetas.append((np.pi/2-i[0])/(np.pi/2-i[1]))
        fig, ax = plt.subplots()
        ax.plot(rs, divided_thetas)
        ax.set(xlabel = 'r ($R_j$)', ylabel = r'$ \theta_c / \theta_m$', 
        title ='Difference Between magnetic and centrifugal equator \n dependence on difference from planet at $\u03BB_{{III}}$ longitude {}{}'.format(phi, u"\N{DEGREE SIGN}")),
        #ylim = (0.9,1))
        ax.grid(which = 'both')
        plt.show()

    def density_topdown_contour(self, gridsize = 30):
        theta = np.pi/2
        gridx, gridy = self.makegrid_2d_negatives(200 ,gridsize= gridsize)

        #print(x_s, y_s)

        ns = []
        for i in range(len(gridx)):
            print('new row, {} to go'.format(len(gridx)-i))
            ns_row = [] 
            for j in range(len(gridx)):
                
                x = gridx[i][j]
                y = gridy[i][j]
                r = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y,x)
                phi_lh = 2*np.pi - phi
                #print(r)
                if r <6:
                    ns_row.append(1e7)
                    continue
                n = self.density_combined(r, theta, phi_lh)
                ns_row.append(n)

            ns.append(ns_row)

        ns_cm = np.array(ns)/1e6

        #log_vas_km = np.log(Vas_km)
        fig, ax = plt.subplots(figsize = (25,15))
        lev_exp = np.arange(np.floor(np.log10(np.amin(ns_cm))-1), np.ceil(np.log10(np.amax(ns_cm))+1), step = 0.25)
        levs = np.power(10, lev_exp)
        cont = ax.contourf(gridx, gridy, ns_cm, cmap = 'bone', levels = levs, norm=mcolors.LogNorm())#, locator=ticker.LogLocator())
        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        ax.add_patch(Circle((0,0), 6, color='c', zorder=2, label = "Io Orbital Radius", fill = False))
        ax.legend()
        ax.set_xlim(-gridsize,gridsize)
        ax.set_ylim(-gridsize,gridsize)
        degrees = theta * 180 /np.pi
        ax.set(xlabel = 'X $(R_J)$', ylabel = 'Y $(R_J)$') #, title = 'Density in the Equatorial plane \n {}')
        fig.colorbar(cont, label = 'Number Density (cm$^{-3}$)')
        ax.set_aspect('equal', adjustable = 'box')
        for r in np.arange(0, 115, 5):
            ax.add_patch(Circle((0,0), r, fill = False, color = 'firebrick', zorder=3))
        ###plt.savefig('images-24-jan-update/va_topdown.png')
        plt.show() 
    def alfven_meridian_slice(self, phi_lh, gridsize = 30, field_line_r = 10, field_line = 'on', within_6 = 'on', num = 200, one_way = 'off'):

        phi_lh_rad = phi_lh*np.pi/180
        phi_rh_rad = 2*np.pi/2 - phi_lh_rad
        
        densities = []
        vas = []
        grids, gridz = self.makegrid_2d_negatives(200 ,gridsize= gridsize)

        r_cent_points = np.linspace(-30, 30, num=num)
        cent_plot_points = []
        mag_plot_points = []
        r_centtheta_magtheta_dict = {}

        for point in r_cent_points:
            if point > 0:
                phi = phi_rh_rad + np.pi 
                phi_lh_for_calc = phi_lh_rad + np.pi
            else: 
                phi = phi_rh_rad
                phi_lh_for_calc = phi_lh_rad
            if -1 < point <1: 
                continue 

            theta_mag_colat = self.complex_mag_equator(abs(point),  phi_lh_for_calc)
            theta_mag = np.pi/2 - (theta_mag_colat -np.pi/2)
            latitude_cent = self.centrifugal_equator(abs(point), phi)
            theta_cent = np.pi/2 - latitude_cent
            r_centtheta_magtheta_dict[point] = [theta_cent, theta_mag]
            z_cent = abs(point) * np.cos(theta_cent)
            z_mag = abs(point) * np.cos(theta_mag)
            mag_plot_points.append([point, z_mag]) 
            cent_plot_points.append([point, z_cent]) 
        cent_plot_points = np.array(cent_plot_points)
        mag_plot_points = np.array(mag_plot_points)

        mag_plot_points_t = np.transpose(mag_plot_points)
        cent_plot_points_t = np.transpose(cent_plot_points)
        spin_eq_plot = np.array([[-30,0], [30,0]])
        spin_eq_plot_t = np.transpose(spin_eq_plot)
        
        for i in range(len(gridz)):
            print('new row, {} to go'.format(len(gridz)-i))
            Vas_row = []
            density_row = []
            for j in range(len(grids)):
                z = gridz[i][j]
                s = grids[i][j]
                r = np.sqrt(z**2 + s**2)
                phi = phi_rh_rad 
                theta = np.arctan2(s,z)
                if r < 6:

                    ''' i think this is causing problems '''
                    
                    if within_6 == 'on':
                        n_at_6 = self.density_combined(r, theta, phi_lh_rad)
                        n = self.density_within_6(r, theta, phi_lh_rad, n_at_6)
                        density_row.append(n)
                        B_overall = self.mag_field_at_point(r, theta, phi)
                        B_x, B_y, B_z = self.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi)
                        B = np.array([B_x, B_y, B_z])
                        B =  B/(10**9)  #chris's code is in nT
                        va = self.calculator(B, n)
                        va_corrected = self.relativistic_correction(va)
                        Vas_row.append(va_corrected)
                        filled = False
                    else:
            
                        Vas_row.append(1e6)
                        Filled = True
                    
                    continue

                
                n = self.density_combined(r, theta, phi_lh_rad)
                density_row.append(n)
                B_overall = self.mag_field_at_point(r, theta, phi)
                B_x, B_y, B_z = self.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi)
                B = np.array([B_x, B_y, B_z])
                B =  B/(10**9)  #chris's code is in nT
                va = self.calculator(B, n)
                va_corrected = self.relativistic_correction(va)
                Vas_row.append(va_corrected)
            densities.append(density_row)
            vas.append(Vas_row)
        
        Vas_km = np.array(vas)/(1000)
        vas_km_clip = np.clip(Vas_km, 10, 3e5)
        vas_km_edits = np.nan_to_num(vas_km_clip, 10)
        fig, ax = plt.subplots(figsize = (25,16))
        #lev_exp = np.arange(np.floor(np.log10(vas_km_edits.min())-1), np.ceil(np.log10(vas_km_edits.max())+1), step = 0.25)
        #levs = np.power(10, lev_exp)
        cont = ax.contourf(grids, gridz, vas_km_edits, cmap = 'bone') #,levels = levs, norm=mcolors.LogNorm())#, locator=ticker.LogLocator()) #, levels = 14)
        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        ax.add_patch(Circle((0,0), 6, color='c', zorder=90, label = "Io Orbital Radius", fill = filed))
        ax.text(0.95, 0.01, 'SYS III (LH) Longitutude = {:.1f}{} '.format(phi_lh, u"\N{DEGREE SIGN}"),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='w', fontsize=16)
        if phi_lh + 180 > 360:
            text_degree = phi_lh - 180
        else:
            text_degree = phi_lh + 180
        ax.text(0.05, 0.99, 'SYS III (LH) Longitutude = {:.1f}{} '.format(text_degree, u"\N{DEGREE SIGN}"),
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes,
        color='w', fontsize=16)
        ax.text(0.05, 0.05, 'CML 201 $\u03BB_{III}$',
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes,
        color='w', fontsize=16)
        plot_results = self.trace_magnetic_field(starting_cordinates=[field_line_r*Rj, np.pi/2 ,phi_lh_rad], one_way=one_way, break_point=2, step = 0.001)
        points = np.array(plot_results[0])
        plottable_list = np.transpose(points)
        plottable_list_rj = plottable_list/Rj
        xs = plottable_list_rj[0]
        ys = plottable_list_rj[1]
        zs = plottable_list_rj[2]
        ss = []
        for i in range(len(xs)):
            phi = np.arctan2(ys[i],xs[i]) 
            ss.append(np.sqrt(xs[i]**2 + ys[i]**2)) #* np.cos(phi - phi_lh_rad )) #np.cos(phi - phi_rh_rad )
        ax.plot(ss,zs, label = 'Field Line')

        ax.set_xlim(-30,30)
        ax.set_ylim(-15,15)
        ax.set(xlabel = ' x($R_J$) \n', ylabel = 'y ($R_J$)', title = 'Density Contour Plot for Given longitude') #, title = 'CML 202 $\u03BB_{III}$')
        if self.aligned == 'no':
            ax.plot(mag_plot_points_t[0], mag_plot_points_t[1], label = 'Magnetic Equator', color = 'm')
            ax.plot(cent_plot_points_t[0], cent_plot_points_t[1], label = 'Centrifugal Equator')
            label = 'Spin Equator'
        if self.aligned == 'yes':
            label = 'Spin & Centrifugal Equator'
        ax.plot(spin_eq_plot_t[0], spin_eq_plot_t[1], label = label)
        fig.colorbar(cont, label = 'Va $(Kms^{-1})$')#, ticks = levs)
        ax.set_aspect('equal', adjustable = 'box')
        ax.legend()
        ###plt.savefig('images-24-jan-update/density_longitude_slice-w-options.png')
        plt.show()
    def density_contour_meridian(self, phi_lh, gridsize =30,field_line_r = 10, field_line = 'on', within_6 = 'on', num = 200, one_way = 'off', show = 'on'):
        phi_lh_rad = phi_lh*np.pi/180
        phi_rh_rad = 2*np.pi/2 - phi_lh_rad
        
        densities = []
        grids, gridz = self.makegrid_2d_negatives(200 ,gridsize= gridsize)

        r_cent_points = np.linspace(-30, 30, num=num)
        cent_plot_points = []
        mag_plot_points = []
        r_centtheta_magtheta_dict = {}

        for point in r_cent_points:
            if point > 0:
                phi = phi_rh_rad + np.pi 
                phi_lh_for_calc = phi_lh_rad + np.pi
            else: 
                phi = phi_rh_rad
                phi_lh_for_calc = phi_lh_rad
            if -1 < point <1: 
                continue 

            theta_mag_colat = self.complex_mag_equator(abs(point),  phi_lh_for_calc)
            theta_mag = np.pi/2 - (theta_mag_colat -np.pi/2)
            latitude_cent = self.centrifugal_equator(abs(point), phi)
            theta_cent = np.pi/2 - latitude_cent
            r_centtheta_magtheta_dict[point] = [theta_cent, theta_mag]
            z_cent = abs(point) * np.cos(theta_cent)
            z_mag = abs(point) * np.cos(theta_mag)
            mag_plot_points.append([point, z_mag]) 
            cent_plot_points.append([point, z_cent]) 
        cent_plot_points = np.array(cent_plot_points)
        mag_plot_points = np.array(mag_plot_points)

        mag_plot_points_t = np.transpose(mag_plot_points)
        cent_plot_points_t = np.transpose(cent_plot_points)
        spin_eq_plot = np.array([[-30,0], [30,0]])
        spin_eq_plot_t = np.transpose(spin_eq_plot)
        
        for i in range(len(gridz)):
            print('new row, {} to go'.format(len(gridz)-i))
            density_row = []
            for j in range(len(grids)):
                z = gridz[i][j]
                s = grids[i][j]
                r = np.sqrt(z**2 + s**2)
                phi = phi_rh_rad
                theta = np.arctan2(s,z)
                if r < 6:
                    n_at_6 = self.density_combined(6, theta, phi_lh_rad)
                    n = self.density_within_6(r, theta, phi_lh_rad, n_at_6)
                    density_row.append(n)
                    continue

                
                n = self.density_combined(r, theta, phi_lh_rad)
                density_row.append(n)
            densities.append(density_row)
        
        densities_cm = np.array(densities)/1e6
        densities_cm_edits = np.clip(densities_cm, 1e-2, 1e10)
        fig, ax = plt.subplots(figsize = (25,16))
        lev_exp = np.arange(np.floor(np.log10(np.amin(densities_cm_edits))-1), np.ceil(np.log10(np.amax(densities_cm_edits))+1), step = 0.25)
        levs = np.power(10, lev_exp)
        #print(levs, levs.size)
        if show == 'on':
            cont = ax.contourf(grids, gridz, densities_cm_edits, cmap = 'bone', levels = levs, norm=mcolors.LogNorm())#, locator=ticker.LogLocator()) #, levels = 14)
            ax.add_patch(Circle((0,0), 6, color='c', zorder=90, label = "R = $6R_J$", fill = False))
        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        
        ax.text(1.00, 0.01, 'SYS III (LH) Longitutude = {:.1f}{} '.format(phi_lh, u"\N{DEGREE SIGN}"),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='w', fontsize=16)
        if phi_lh + 180 > 360:
            text_degree = phi_lh - 180
        else:
            text_degree = phi_lh + 180
        ax.text(0.05, 0.99, 'SYS III (LH) Longitutude = {:.1f}{} '.format(text_degree, u"\N{DEGREE SIGN}"),
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes,
        color='w', fontsize=16)
        #ax.text(0.05, 0.05, 'CML 201 $\u03BB_{III}$',
        #verticalalignment='top', horizontalalignment='left',
        #transform=ax.transAxes,
        #color='w', fontsize=16)
        if field_line == 'on':
            plot_results = self.trace_magnetic_field(starting_cordinates=[field_line_r*Rj, np.pi/2 ,phi_lh_rad], one_way=one_way, break_point=2, step = 0.001)
            points = np.array(plot_results[0])
            plottable_list = np.transpose(points)
            plottable_list_rj = plottable_list/Rj
            xs = plottable_list_rj[0]
            ys = plottable_list_rj[1]
            zs = plottable_list_rj[2]
            ss = []
            for i in range(len(xs)):
                phi = np.arctan2(ys[i],xs[i]) 
                ss.append(np.sqrt(xs[i]**2 + ys[i]**2)) #* np.cos(phi - phi_lh_rad )) #np.cos(phi - phi_rh_rad )
            ax.plot(ss,zs, label = 'Field Line')

        ax.set_xlim(-30,30)
        ax.set_ylim(-15,15)
        ax.set(xlabel = ' X($R_J$) \n', ylabel = 'Z ($R_J$)')#, title = 'Density Contour Plot for Given longitude') #, title = 'CML 202 $\u03BB_{III}$')
        ax.plot(mag_plot_points_t[0], mag_plot_points_t[1], label = 'Magnetic Equator', color = 'm')
        if self.aligned == 'no':
            
            ax.plot(cent_plot_points_t[0], cent_plot_points_t[1], label = 'Centrifugal Equator')
            label = 'Spin Equator'
        elif self.aligned == 'yes' and self.model =='dipole':
            label = 'Spin, Centrifugal & Magnetic Equator'
        else:
            label = 'Spin & Centrifugal Equator'
        ax.plot(spin_eq_plot_t[0], spin_eq_plot_t[1], label = label)
        ax.grid(False)
        plt.tick_params(axis='both', which='both', bottom='off', top='off')#, labelbottom='off', right='off', left='off', labelleft='off')
        if show == 'on':
            cbar = fig.colorbar(cont, label = 'Density $(cm^{-3})$')#, ticks = levs)
            cbar.ax.set_yticklabels(levs)
        ax.set_aspect('equal', adjustable = 'box')
        ax.legend()
        ###plt.savefig('images-24-jan-update/density_longitude_slice-w-options.png')
        plt.show()
        return r_centtheta_magtheta_dict, grids, gridz, densities_cm_edits, levs, mag_plot_points_t, cent_plot_points_t, spin_eq_plot_t
    
    def phippsbagfig_recreate(self):
        phi_rh_degs = np.array([39,69,99,159,339])
        phi_lh_degs = 360 - phi_rh_degs
        rs = np.linspace(2,50, num = 100) 
        thetas_phis = {}
        phi_rh_rads = phi_rh_degs * np.pi/180
        phi_lh_rads = phi_lh_degs * np.pi/180
        for phi in phi_rh_rads:
            thetas = []
            for r in rs:
                theta = self.centrifugal_equator(r, phi)
                #print(phi, theta)
                thetas.append(theta * 180/np.pi)
            thetas_phis[phi] = thetas
        fig, ax = plt.subplots()
        for key in thetas_phis:
            ax.plot(rs, thetas_phis[key], label = ' $\u03BB_{{III}}$ = {:.1f}'.format(360 - key * 180/np.pi))
        labelLines(ax.get_lines(), zorder=2.5)

        ax.grid()
        #ax.legend()
        ax.yaxis.set_ticks_position('both')
        #ax.set(title = 'Radial Distance vs Longitude of centrifugual equator',
        ax.set(xlabel = 'Radial Distance ($R_J$)',ylabel = 'Latitude of Centrifigual Equator (Degrees)')
        ###plt.savefig('images-24-jan-update/phippsbag.png')
        plt.show()

    ''' last but not least, alfven functions '''
    def calculator(self, B, n):
        magB = np.linalg.norm(B)
        rho = n * self.avgIonMass 
        Va = magB/np.sqrt((rho * mu_0))
        #print('b = {}, va = {}, rho = {}, magB = {}'.format(B, Va, rho, magB))
        return Va

    def alfven_topdown_equatorial_plane(self, gridsize = 30):
        theta = np.pi/2 #<- CHANGE THIS TO VIEW A SLIGHTLY DIFFERENT PLANE
        gridx, gridy = self.makegrid_2d_negatives(200 ,gridsize= gridsize)
        n_0s = []
        #print(x_s, y_s)

        Vas = []
        for i in range(len(gridx)):
            print('new row, {} to go'.format(len(gridx)-i))
            Vas_row = [] 
            for j in range(len(gridx)):
                
                x = gridx[i][j]
                y = gridy[i][j]
                r = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y,x)
                phi_lh = 2*np.pi - phi
                #print(r)
                if r <6:
                    ''' 
                    the stuff below, whilst technically correct, leads to a va way higher than anywhere else. This ruins the look of the contour 

                    n_at_6 = self.density_combined(r, theta, phi_lh)
                    n = self.density_within_6(r, theta, phi_lh, n_at_6)
                    B_overall = self.mag_field_at_point(r, theta, phi)
                    B_x, B_y, B_z = self.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi)
                    B = np.array([B_x, B_y, B_z])
                    va = self.calculator(B, n)
                    va_corrected = self.relativistic_correction(va)
                    Vas_row.append(va_corrected)
                    '''
                    Vas_row.append(1e4)
                    continue
                n = self.density_combined(r, theta, phi_lh)
                phi = np.arctan2(y,x)
                B_overall = self.mag_field_at_point(r, theta, phi)
                B_x, B_y, B_z = self.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi)
                B = np.array([B_x, B_y, B_z])
                B =  B/(10**9)  #chris's code is in nT
                va = self.calculator(B, n)
                va_corrected = self.relativistic_correction(va)
                Vas_row.append(va_corrected)


            Vas.append(Vas_row)

        Vas_km = np.array(Vas)/(1000)
        lev_exp = np.arange(np.floor(np.log10(np.amin(Vas_km))-1), np.ceil(np.log10(np.amax(Vas_km))+1), step = 0.25)
        #print(lev_exp)
        levs = np.power(10, lev_exp)
        print(lev_exp, levs)
        #log_vas_km = np.log(Vas_km)
        fig, ax = plt.subplots(figsize = (25,15))
        cont = ax.contourf(gridx, gridy, Vas_km, cmap = 'bone', norm = mcolors.LogNorm(), levels = levs)#, locator=ticker.LogLocator())
        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        ax.add_patch(Circle((0,0), 6, color='c', zorder=2, label = "Io Orbital Radius"))
        ax.legend()
        ax.set_xlim(-gridsize,gridsize)
        ax.set_ylim(-gridsize,gridsize)
        degrees = theta * 180 /np.pi
        ax.set(xlabel = 'X $(R_J)$', ylabel = 'Y $(R_J)$')#, title = 'Alfven velocity in the Equatorial plane')
        ax.grid(False)
        cbar = fig.colorbar(cont, label = '$V_a (kms^{-1})$', ticks = [1,10,100,1000,10000])

        ax.set_aspect('equal', adjustable = 'box')
        for r in np.arange(0, 115, 5):
            ax.add_patch(Circle((0,0), r, fill = False, color = 'firebrick', zorder = 3))
        ###plt.savefig('images-24-jan-update/va_topdown.png')
        plt.show() 
        return Vas
    def alfven_meridian_slice(self, phi_lh, gridsize = 30, field_line_r = 10, field_line = 'on', within_6 = 'on', num = 200, one_way = 'off'):

        phi_lh_rad = phi_lh*np.pi/180
        phi_rh_rad = 2*np.pi/2 - phi_lh_rad
        
        densities = []
        vas = []
        grids, gridz = self.makegrid_2d_negatives(200 ,gridsize= gridsize)

        r_cent_points = np.linspace(-30, 30, num=num)
        cent_plot_points = []
        mag_plot_points = []
        r_centtheta_magtheta_dict = {}

        for point in r_cent_points:
            if point > 0:
                phi = phi_rh_rad + np.pi 
                phi_lh_for_calc = phi_lh_rad + np.pi
            else: 
                phi = phi_rh_rad
                phi_lh_for_calc = phi_lh_rad
            if -1 < point <1: 
                continue 

            theta_mag_colat = self.complex_mag_equator(abs(point),  phi_lh_for_calc)
            theta_mag = np.pi/2 - (theta_mag_colat -np.pi/2)
            latitude_cent = self.centrifugal_equator(abs(point), phi)
            theta_cent = np.pi/2 - latitude_cent
            r_centtheta_magtheta_dict[point] = [theta_cent, theta_mag]
            z_cent = abs(point) * np.cos(theta_cent)
            z_mag = abs(point) * np.cos(theta_mag)
            mag_plot_points.append([point, z_mag]) 
            cent_plot_points.append([point, z_cent]) 
        cent_plot_points = np.array(cent_plot_points)
        mag_plot_points = np.array(mag_plot_points)

        mag_plot_points_t = np.transpose(mag_plot_points)
        cent_plot_points_t = np.transpose(cent_plot_points)
        spin_eq_plot = np.array([[-30,0], [30,0]])
        spin_eq_plot_t = np.transpose(spin_eq_plot)
        
        for i in range(len(gridz)):
            print('new row, {} to go'.format(len(gridz)-i))
            Vas_row = []
            density_row = []
            for j in range(len(grids)):
                z = gridz[i][j]
                s = grids[i][j]
                r = np.sqrt(z**2 + s**2)
                phi = phi_rh_rad 
                theta = np.arctan2(s,z)
                if r < 6:

                    ''' i think this is causing problems '''
                    
                    if within_6 == 'on':
                        n_at_6 = self.density_combined(r, theta, phi_lh_rad)
                        n = self.density_within_6(r, theta, phi_lh_rad, n_at_6)
                        density_row.append(n)
                        B_overall = self.mag_field_at_point(r, theta, phi)
                        B_x, B_y, B_z = self.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi)
                        B = np.array([B_x, B_y, B_z])
                        B =  B/(10**9)  #chris's code is in nT
                        va = self.calculator(B, n)
                        va_corrected = self.relativistic_correction(va)
                        Vas_row.append(va_corrected)
                        filled = False
                    else:
            
                        Vas_row.append(1e6)
                        filled = True
                    
                    continue

                
                n = self.density_combined(r, theta, phi_lh_rad)
                density_row.append(n)
                B_overall = self.mag_field_at_point(r, theta, phi)
                B_x, B_y, B_z = self.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi_rh_rad)
                B = np.array([B_x, B_y, B_z])
                B =  B/(10**9)  #chris's code is in nT
                va = self.calculator(B, n)
                va_corrected = self.relativistic_correction(va)
                Vas_row.append(va_corrected)
            densities.append(density_row)
            vas.append(Vas_row)
        
        Vas_km = np.array(vas)/(1000)
        vas_km_clip = np.clip(Vas_km, 10, 3e5)
        vas_km_edits = np.nan_to_num(vas_km_clip, 10)
        fig, ax = plt.subplots(figsize = (25,16))
        #print(np.log10(np.amin(vas_km_edits))-1 ,np.log10(np.amax(vas_km_edits))+1)
        lev_exp = np.arange(np.floor(np.log10(np.amin(vas_km_edits))-1), np.ceil(np.log10(np.amax(vas_km_edits))+1), step = 0.25)
        #print(lev_exp)
        levs = np.power(10, lev_exp)
        #print(levs)
        cont = ax.contourf(grids, gridz, vas_km_edits, cmap = 'bone', levels = levs, norm=mcolors.LogNorm())#, locator=ticker.LogLocator()) #, levels = 14)
        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        ax.add_patch(Circle((0,0), 6, color='c', zorder=90, label = "Io Orbital Radius", fill = filled))
        ax.text(0.95, 0.01, 'SYS III (LH) Longitutude = {:.1f}{} '.format(phi_lh, u"\N{DEGREE SIGN}"),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='w', fontsize=16)
        if phi_lh + 180 > 360:
            text_degree = phi_lh - 180
        else:
            text_degree = phi_lh + 180
        ax.text(0.05, 0.99, 'SYS III (LH) Longitutude = {:.1f}{} '.format(text_degree, u"\N{DEGREE SIGN}"),
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes,
        color='w', fontsize=16)
        ax.text(0.05, 0.05, 'CML 201 $\u03BB_{III}$',
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes,
        color='w', fontsize=16)
        if field_line == 'on':
            plot_results = self.trace_magnetic_field(starting_cordinates=[field_line_r*Rj, np.pi/2 ,phi_lh_rad], one_way=one_way, break_point=2, step = 0.001)
            points = np.array(plot_results[0])
            plottable_list = np.transpose(points)
            plottable_list_rj = plottable_list/Rj
            xs = plottable_list_rj[0]
            ys = plottable_list_rj[1]
            zs = plottable_list_rj[2]
            ss = []
            for i in range(len(xs)):
                phi = np.arctan2(ys[i],xs[i]) 
                ss.append(np.sqrt(xs[i]**2 + ys[i]**2)) #* np.cos(phi - phi_lh_rad )) #np.cos(phi - phi_rh_rad )
            ax.plot(ss,zs, label = 'Field Line')

        ax.set_xlim(-30,30)
        ax.set_ylim(-15,15)
        ax.set(xlabel = ' x($R_J$) \n', ylabel = 'y ($R_J$)', title = 'Density Contour Plot for Given longitude') #, title = 'CML 202 $\u03BB_{III}$')
        if self.aligned == 'no':
            ax.plot(mag_plot_points_t[0], mag_plot_points_t[1], label = 'Magnetic Equator', color = 'm')
            ax.plot(cent_plot_points_t[0], cent_plot_points_t[1], label = 'Centrifugal Equator')
            label = 'Spin Equator'
        if self.aligned == 'yes':
            label = 'Spin & Centrifugal Equator'
        ax.plot(spin_eq_plot_t[0], spin_eq_plot_t[1], label = label)
        ticks = levs[::4]
        cbar = plt.colorbar(cont, label = 'Va $(Kms^{-1})$', ticks = ticks, format = ticker.FixedFormatter(seq =['$10^1$', '$10^2$', '$10^3$','$10^4$','$10^5$','$10^6$']) )#, ticks = [1e1,1e2,1e3,1e4,1e5,1e6])#, ticks = levs)
        cbar.set_ticks = ticks
        cbar.set_yticklabels = ['$10^1$', '$10^2$', '$10^3$','$10^4$','$10^5$','$10^6$']
        ax.set_aspect('equal', adjustable = 'box')
        ax.legend()
        print(ticks)
        ###plt.savefig('images-24-jan-update/density_longitude_slice-w-options.png')
        plt.show()
    
        
    def travel_time(self, startpoint = [10, np.pi/2, 212* np.pi/180], direction =   'forward', path_plot = 'off', dr_plot = 'off', print_time = 'on',
     va_plot ='off',b_plot = 'off', n_plot = 'off', vvsrplot = 'off', uncorrected_vvsr = 'off',
    debug_plot = 'off', nvsrplot = 'off', break_point = 2):
        '''
        Calculate the travel path/time of an alfven wave from a given startpoint to the ionosphere. 
        input startpoint [r, theta, phi] where r is in rj and phi is left handed in RADIANS
        direction input - 'forward' travels along the magnetic field vector to the southern hemisphere, backward is the reverse 

        ''' 
        saving_start = deepcopy(startpoint)
        startpoint[0] = startpoint[0]*Rj
        #phi_lh = startpoint[2]


        plot_results = self.trace_magnetic_field(starting_cordinates=startpoint, one_way='on', break_point=break_point, step = 0.001, pathing= direction)
        
        points = np.array(plot_results[0])
        #print(len(points))
        drs = np.array(plot_results[3])
        rs = np.array(plot_results[4])
        drs_km = drs/1e3
        rs_km = rs/1e3
        rs_rj = rs/Rj
        rs_rj_popped =  rs_rj[rs_rj > 6]
        Bs = np.array(plot_results[2]) / 1e9 # turn nano tesla into T
        B_along_path = []
        n_along_path = []
        ''' 
        this returns the path taken (in terms of point by point) taken by the alfven wave (points)
        and the magnetic field at each points (Bs)
        '''  
        time = 0 
        va_uncorrected_list = []
        va_corrected_list = []
        time_uncorrected = 0
        

        for i in range(len(points)-1):
            start_point = points[i]
            end_point = points[i+1]
            difference = end_point - start_point
            distance = np.linalg.norm(difference)
            #distance = np.sqrt( ((end_point[0] - start_point[0])**2) + ((end_point[1] - start_point[1])**2) + ((end_point[2] - start_point[2])**2) )
            
            midpoint = end_point - difference/2
            
            B_start = Bs[i]
            B_end = Bs[i+1]
            magB_start = np.linalg.norm(B_start)
            magB_end = np.linalg.norm(B_end)
            averageB = (magB_end + magB_start)/2
            B_along_path.append(averageB)
            

            r, theta, phi = self.cart_to_sph(midpoint[0], midpoint[1], midpoint[2])
            r = r/Rj
            phi_lh = 2*np.pi - phi
            
            if r < 6: 
                n_at_6 = self.density_combined(r, theta, phi_lh)
                n = self.density_within_6(r, theta, phi_lh, n_at_6)
                n_along_path.append(n)
                va = self.calculator(averageB, n)
                va_uncorrected_list.append(va)
                va_corrected = self.relativistic_correction(va)
                va_corrected_list.append(va_corrected)
                traveltime = distance/va_corrected
                time += traveltime
                continue
            n = self.density_combined(r, theta, phi_lh)
            
            n_along_path.append(n)
            va = self.calculator(averageB, n)

            va_uncorrected_list.append(va)

            va_corrected = self.relativistic_correction(va)
            va_corrected_list.append(va_corrected)
            
            traveltime_uncorrected = distance/va
            traveltime = distance/va_corrected
            #print('start Point {} \nEnd Point {} \nMidpoint {} \nDifference {} \nDistance {} \nva {} \nTravel time{}\n \n '.format(start_point, end_point, midpoint, difference, distance, va_corrected, traveltime ))
            time += traveltime
            time_uncorrected += traveltime_uncorrected
        
        if print_time == 'on':
            print('travel time = {:.2f}s (= {:.1f}mins)'.format(time, time/60))
            print('uncorrected travel time = {:.2f}s (= {:.1f}mins)'.format(time_uncorrected, time_uncorrected/60))
        '''
        As the travel time seems to be a bit weird, good idea to be plot the path etc 
        '''
        plottable_list = np.transpose(points)
        if path_plot == 'on':
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
            fig.suptitle('Field Line for which travel time was calculated using {}'.format(self.model))
            ax.set_title('Start Point = ({:.0f}, {:.1f}{}, {:.1f}{})SYSIII'.format(startpoint[0]/Rj, startpoint[1] * 180/np.pi, u"\N{DEGREE SIGN}", phi_lh * 180/np.pi, u"\N{DEGREE SIGN}"))
            #ax.text(1,2,40, 'time = {:.1f}s (= {:.1f}mins)'.format(time, time/60))
            ax.text2D(0.05, 0.95, 'time = {:.1f}s (= {:.1f}mins)'.format(time, time/60), transform=ax.transAxes)

            #plt.legend()
            ###plt.savefig('images-24-jan-update/travel_time_trace.png'.format(self.model))
            plt.show()
        if dr_plot == 'on':

            fig, ax1 = plt.subplots()
            numbers = list(range(len(drs_km)))
            drs_rj = np.array(drs)/Rj
            
            ax1.plot(numbers, drs_rj, label = 'Distance Between Points ($R_j$)', color = 'k')
            ax1.set_xlabel('Point Index')
            ax1.set_ylabel('Distance Between Points ($R_j$)', color = 'k')
            ax1.tick_params(axis='y', labelcolor='k')
            #ax1.legend(loc=0)

            ax2 = ax1.twinx() 
            ax2.plot(numbers, rs_rj, label = 'r ($R_J$)', color = 'c', linestyle ='--')
            ax2.set_ylabel('Distance From Planet ($R_J$)', color = 'c')
            ax2.tick_params(axis='y', labelcolor='c')
            #ax2.legend(loc = 1)
            #plt.legend()
            plt.show()
        if va_plot == 'on':
            fig, ax1 = plt.subplots()
            numbers = list(range(len(va_corrected_list)))
            #numbers_r = list(range(len(rs_rj_popped)))
            numbers_r = list(range(len(rs_rj)))
            
            ax1.plot(numbers, va_corrected_list, label = '$v_A$ corrected ($ms^{-1}$)', color = 'c')
            ax1.set_xlabel('Point Index')
            ax1.set_ylabel('Speed ($ms^{-1}$)', color = 'b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.plot(numbers, va_uncorrected_list, label = '$v_A$ uncorrected ($ms^{-1}$)', color = 'b', linestyle ='--')
            #ax1.legend()
            ax2 = ax1.twinx() 
            #ax2.plot(numbers_r, rs_rj_popped, label = 'Distance from planet ($R_J$)', color = 'r', linestyle ='-')
            ax2.plot(numbers_r, rs_rj, label = 'Distance from planet ($R_J$)', color = 'r', linestyle ='-')
            ax2.set_ylabel('S ($R_J$)', color = 'r')
            ax2.tick_params(axis='y', labelcolor='r')
            #ax2.legend()
            #plt.grid(True)
            plt.figlegend()
            fig.suptitle('Effect of including relativistic correction')
            ###plt.savefig('images-24-jan-update/va correction effects.png')
            plt.grid(which='both')
            plt.show()
        if b_plot == 'on':
            fig, ax1 = plt.subplots()
            numbers_b = list(range(len(B_along_path)))
            B_along_path_nt = np.array(B_along_path) * 1e9
            ax1.plot(numbers_b, B_along_path_nt, label = 'B along path', color = 'k')
            ax1.set_xlabel('Point Index')
            ax1.set_ylabel('B Along Path', color = 'k')
            ax1.tick_params(axis='y', labelcolor='k')
            #ax1.legend(loc=0)
            numbers_r = list(range(len(rs_rj)))
            ax2 = ax1.twinx() 
            ax2.plot(numbers_r, rs_rj, label = 'S ($R_J$)', color = 'c', linestyle ='--')
            ax2.set_ylabel('Distance From Planet ($R_J$)', color = 'c')
            ax2.tick_params(axis='y', labelcolor='c')
            #ax2.legend(loc = 1)
            #plt.legend()
            plt.grid()
            plt.show()
        if n_plot == 'on':
            fig, ax1 = plt.subplots()
            numbers_n = list(range(len(n_along_path)))
            
            ax1.plot(numbers_n, n_along_path, label = 'Number Density Along Path', color = 'k')
            ax1.set_xlabel('Point Index')
            ax1.set_ylabel('N Along Path', color = 'k')
            ax1.tick_params(axis='y', labelcolor='k')
            #ax1.legend(loc=0)
            numbers_r = list(range(len(rs_rj)))
            ax2 = ax1.twinx() 
            ax2.plot(numbers_r, rs_rj, label = 'r ($km$)', color = 'c', linestyle ='--')
            ax2.set_ylabel('Distance From Planet (km)', color = 'c')
            ax2.tick_params(axis='y', labelcolor='c')
            #ax2.legend(loc = 1)
            #plt.legend()
            plt.show()
        if debug_plot == 'on':
            fig, ax1 = plt.subplots()
            numbers_b = list(range(len(B_along_path)))
            
            ax1.plot(numbers_b, B_along_path, label = 'B along path', color = 'k')
            ax1.set_xlabel('Point Index')
            ax1.set_ylabel('B Along Path', color = 'k')
            ax1.tick_params(axis='y', labelcolor='k')
            #ax1.legend(loc=0)
            ax2 = ax1.twinx() 
            numbers_n = list(range(len(n_along_path)))
            ax2.plot(numbers_n, n_along_path, label = 'n along path', color = 'b')
            ax2.set_xlabel('Point Index')
            ax2.set_ylabel('N Along Path', color = 'b')
            ax2.tick_params(axis='y', labelcolor='b')
            #ax2.legend(loc = 1)
            #plt.legend()
            plt.show()
        if nvsrplot == 'on':
            fig, ax = plt.subplots()
            ns_cm = np.array(n_along_path)/1e6

            
            ax.plot(rs_rj[:-1], ns_cm, label = 'Density Along Alfven Wave Path ($cm^{-3}$)', color = 'k')
            ax.set_xlabel('r ($R_j$)')
            ax.set_ylabel('n Along Path', color = 'k')
            
            if direction == 'forward':
                endpoint = 'South'
            else:
                endpoint = 'North'
            ax.set_title('Effect of distance from Jupiter on Density along path taken by Alfven wave \n From $\u03BB_{{III}}$ ({:.0f},{:.0f}{},{:.0f}{}) to: {} Hemsiphere'.format(startpoint[0]/Rj, 180*startpoint[1]/np.pi, u"\N{DEGREE SIGN}"
             ,180*saving_start[2]/np.pi, u"\N{DEGREE SIGN}",endpoint))
            ax.grid(which = 'both')
            plt.axvline(x=6, label = 'Orbit of Io', color = 'red', linestyle = 'dashed')
            ax.legend()
            plt.show()
        if vvsrplot == 'on':
            fig, ax1 = plt.subplots()
            ax1.plot(rs_rj[:-1], va_corrected_list, label = '$V_A$ corrected ($ms^{-1}$)', color = 'c')
            ax1.set_xlabel('S $(R_J)$')
            ax1.set_ylabel('Alfven Velocity ($ms^{-1}$)', color = 'b')
            ax1.plot(rs_rj[:-1], va_uncorrected_list, label = '$V_A$ uncorrected ($ms^{-1}$)', color = 'b', linestyle ='--')

            ax1.legend()
            ax1.set_title('Alfven Velocity Against Distance from planet \n Including effect of including relativistic correction')
            ###plt.savefig('images-24-jan-update/va vs r.png')
            plt.grid(which='both')
            plt.show()
        if uncorrected_vvsr == 'on':
            fig, ax1 = plt.subplots()
            ax1.plot(rs_rj[:-1], va_corrected_list, label = '$v_A$ corrected ($ms^{-1}$)', color = 'c')
            ax1.set_xlabel('R $(R_J)$')
            ax1.set_ylabel('Alfven Velocity ($ms^{-1}$)', color = 'b')
            #ax1.plot(rs_rj[:-1], va_uncorrected_list, label = '$v_A$ uncorrected ($ms^{-1}$)', color = 'b', linestyle ='--')

            ax1.legend()
            ax1.set_title('Alfven Velocity Against Distance from planet')
            ###plt.savefig('images-24-jan-update/va vs r.png')
            plt.grid(which='both')
            plt.show()
        return time, va_corrected_list, va_uncorrected_list, plottable_list, rs_km, points


    def relativistic_correction(self, va):
        ''' 
        Input va
        return corrected va 
        equation taken from https://www.aanda.org/articles/aa/pdf/2012/06/aa18630-11.pdf eq 2. 
        '''
        corrected_va = (va * 3e8)/np.sqrt(va**2 + (3e8)**2)
        return corrected_va

    def plot_rel_effect(self, both = 'on'):
        calc = self.travel_time()
        points_plottable = calc[3]
        plottable_list_rj = points_plottable/Rj
        va_uncorrected_list = np.array(calc[2])/1e3
        va_corrected_list = np.array(calc[1])/1e3
        numbers = list(range(len(va_corrected_list)))
        rs_km = calc[4]
        rs_km_plot = rs_km[:-1]
        if both =='on':
            fig = plt.figure(figsize=plt.figaspect(.5))
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            x,y, z = self.make_sphere()
            ax.plot_surface(x, y, z, color = 'yellow', zorder=100, label = 'Jupiter')
            ax.set_xlim3d(-40, 40)
            ax.set_ylim3d(-40, 40)
            ax.set_zlim3d(-40, 40)
            ax.set_xlabel('$X, R_j$', fontsize=10)
            ax.set_ylabel('$Y, R_J$', fontsize=10)
            ax.plot(plottable_list_rj[0], plottable_list_rj[1], plottable_list_rj[2],color = 'black', label = 'Field Trace')

            ax2 = fig.add_subplot(1, 2, 2)
            ax2.plot(rs_km_plot, va_corrected_list, label = 'va corrected', color = 'c')

            ax3 = ax2.twinx()

            ax3.plot(rs_km_plot, va_uncorrected_list, label = 'va uncorrected', color = 'r', linestyle = '--')

            plt.show()

    def plot_correction(self):
        c = 3e8
        speeds = np.linspace(0, 10*c, num = 1000)    
        rel_speeds = []
        for speed in speeds:
            rel_speed = self.relativistic_correction(speed)
            rel_speeds.append(rel_speed)    
        normalised_rel_speeds = np.array(rel_speeds)/c
        normalised_speeds = np.array(speeds)/c
        fig, ax = plt.subplots()
        ax.plot(normalised_rel_speeds, normalised_speeds, color = 'b')
        ax.set(ylabel = 'Speed/c', xlabel = 'Corrected Speed/c')#, xlim =(0,1.6*c), ylim =(0,1.6*c))
        plt.suptitle('Effect of Relatavistic Correction on Speed')
        ax.grid()

        plt.show()

    def multiple_travel_times(self, num = 8, plot = 'on', direction = 'forward', r = 15):
        ''' docstring goes here ''' 

        ''' TO START WITH, THIS IS JUST GONNA BE ALL ON ONE PLOT, BUT IT COULD BE EXTENDED TO HAVE THEM ALL ON SEPERATE PLOTS! ''' 
        startingPoints = []
        spacing = 2*np.pi/num
        for n in range(num):
            startingPoints.append([r, np.pi/2, n*spacing])
        if plot == 'on':
            fig = plt.figure()
            ax = fig.gca(projection='3d') # initialise figure
            colours = ['b','g','r','c','m','k',] # just setting this up for use later
            legend_elements = [] #MATPLOTLIB IS A PAIN
            #make the sphere 
            x,y, z = self.make_sphere()
            ax.plot_surface(x, y, z, color = 'yellow', zorder=100, label = 'Jupiter')
            ax.set_xlim3d(-40, 40)
            ax.set_ylim3d(-40, 40)
            ax.set_zlim3d(-40, 40)
            ax.set_xlabel('$X, R_j$', fontsize=10)
            ax.set_ylabel('$Y, R_J$', fontsize=10)
        color_index = 0
        angle_time_dictionary = {}
        for point in startingPoints:
            print('New Startpoint! {} to go'.format(len(startingPoints) - startingPoints.index(point)))
            phi_rh_rad = point[2] 
            if self.aligned == 'no':
                cent_eq_latitude = self.centrifugal_equator(r, phi_rh_rad)
                colatitude = np.pi/2 - cent_eq_latitude
            else:
                colatitude = np.pi/2
            point[2] = 2*np.pi - point[2]
            point[1] = colatitude
            phi_lh = point[2]
            phi_lh_deg = phi_lh * 180/np.pi
            calc = self.travel_time(startpoint=point, print_time='off', direction = direction)
            time = calc[0]
            plot_points = calc[3]
            plot_points_rj = np.array(plot_points)/Rj
            angle_time_dictionary[phi_lh] = time
            if plot =='on':
                label = '$\u03BB_(III)$ = {}{}, time = {:.0f}mins'.format(phi_lh_deg,u"\N{DEGREE SIGN}", time/60)
                ax.plot(plot_points_rj[0], plot_points_rj[1], plot_points_rj[2], label = label ,color = colours[color_index])             
                legend_elements.append(Line2D([0], [0], color=colours[color_index], lw=2, label=label))
                color_index +=1
                if color_index > len(colours)-1:
                    color_index = 0
        
        #print(angle_time_dictionary)
        if plot == 'on':
            
            ax.legend(handles=legend_elements, loc = 'upper left')
            plt.show()
        return angle_time_dictionary

    def plot_angle_vs_time(self, num = 10, direction = "backward", r = 10):
        ''' generate a plot of how the travel time depends with the angle of the starting point. ''' 
        angles_times = self.multiple_travel_times(num=num, plot='off', direction=direction, r=r)
        #print(angles_times)
        angles = list(angles_times.keys())
        times = list(angles_times.values())
        angles_degree = [x*180/np.pi for x in angles]
        times_mins = [x/60 for x in times]
        print(angles_degree, times_mins)
        fig, ax = plt.subplots()
        ax.plot(angles_degree, times_mins)
        if r == 20 and self.aligned == 'yes':
            ax.set_ylim(11,13)
        if r == 6 and self.aligned == 'yes':
            ax.set_ylim(2,4)
        if direction == 'forward':
            endpoint = 'South'
        else:
            endpoint = 'North'
        ax.set(xlabel = '$\u03BB_{III}$ (Degrees)', ylabel = 'Time (Minutes)')
        #title ='Effect of Starting longitude In Equatorial Plane on Travel Time \n From r = {}$R_J$ to Destination: {} Hemsiphere \n {}'.format(r,endpoint, self.plot_label))
        #ax.tick_params(labelright = True)
        #plt.grid(which = 'both')
        plt.show()

    def plot_B_debug_time(self, phi_lh = 69, gridsize = 30):
        grids, gridz = self.makegrid_2d_negatives(200 ,gridsize= gridsize) #CHANGE THIS BACK TO 100 WHEN ITS WORKING
        phi_rh = 360-phi_lh
        phi_lh_rad = phi_lh * np.pi/180
        phi_rh_rad = phi_rh *np.pi/180
        Bs = []
        for i in range(len(gridz)):
            print('new row, {} to go'.format(len(gridz)-i))
            Bs_row = []

            for j in range(len(grids)):
                z = gridz[i][j]
                s = grids[i][j]
                r = np.sqrt(z**2 + s**2)
                if r<6:
                    Bs_row.append(1e-9)
                    continue
                '''
                if s<0:
                    phi = phi_rh_rad + np.pi
                else:
                    phi = phi_rh_rad
                '''
                phi = phi_lh_rad
                theta = np.arctan2(s,z)
                B = self.mag_field_at_point(r,theta,phi)
                B_overall = self.mag_field_at_point(r, theta, phi)
                B_x, B_y, B_z = self.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi_rh)
                B = np.array([B_x, B_y, B_z])
                B = B/(1e9) #CHRIS code outputs nT
                magB = np.linalg.norm(B)
                Bs_row.append(magB)
            Bs.append(Bs_row)
        Bs_plot = np.array(Bs)
        fig, ax = plt.subplots(figsize = (25,16))
        lev_exp = np.arange(np.floor(np.log10(np.min(Bs_plot))-1), np.ceil(np.log10(np.max(Bs_plot))+1), step = 0.25)
        levs = np.power(10, lev_exp)
        cont = ax.contourf(grids, gridz, Bs_plot, cmap = 'bone', levels = levs, norm=mcolors.LogNorm())# locator=ticker.LogLocator())
        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        ax.add_patch(Circle((0,0), 6, color='c', zorder=90, label = "Io Orbital Radius"))
        ax.text(0.95, 0.01, 'SYS III (lh) Longitutude = {} '.format(phi_lh),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='k', fontsize=15)
        ax.set_xlim(-30,30)
        ax.set_ylim(-15,15)
        ax.set(xlabel = '$R_J$ \n', ylabel = '$R_J$', title = 'Meridian Slice')
        fig.colorbar(cont, label = 'B $T$')
        ax.set_aspect('equal', adjustable = 'box')
        ax.legend()
        ###plt.savefig('images-24-jan-update/B side slice.png')
        plt.show() 

    def plot_multiple_distances(self, num = 50, direction = 'backward'):
        overall_distances = []
        phis = []
        startingPoints = []
        spacing = 2*np.pi/num
        for n in range(num):
            startingPoints.append([10, np.pi/2, n*spacing])
            phis.append(n*spacing)
        for point in startingPoints:
            print('New Startpoint = ', point)
            totaldistance = 0
            calc = self.travel_time(startpoint=point, print_time='off', direction = direction)
            path_taken = calc[5]
            for i in range(len(path_taken)-1):
                start_point = path_taken[i]
                end_point = path_taken[i+1]
                difference = end_point - start_point
                distance = np.linalg.norm(difference)
                totaldistance += distance
            overall_distances.append(totaldistance)
        phis_degrees = np.array(phis) * 180/np.pi
        overall_distances_km = np.array(overall_distances) / 1e3
        fig, ax = plt.subplots()
        ax.plot(phis_degrees, overall_distances_km)
        if direction == 'forward':
            endpoint = 'South'
        else:
            endpoint = 'North'
        ax.set(xlabel = 'phi $\u03BB_{III}$ (Degrees)', ylabel = 'Distance (Km)', 
        title ='Effect of Starting longitude In Equatorial Plane on Distance Travelled By Alfven Waves \n Destination: {} Hemsiphere'.format(endpoint))
        ax.grid(which = 'both')
        plt.show()

    def multiple_travel_times_both_directions(self, num = 8, r = 10):
        ''' docstring goes here ''' 

        ''' TO START WITH, THIS IS JUST GONNA BE ALL ON ONE PLOT, BUT IT COULD BE EXTENDED TO HAVE THEM ALL ON SEPERATE PLOTS! ''' 
        startingPoints = []
        spacing = 2*np.pi/num
        for n in range(num):
            startingPoints.append([r, np.pi/2, n*spacing])
        angle_time_dictionary = {}
        for point in startingPoints:
            print('New Startpoint, ', point)
            phi_lh = point[2]
            phi_lh_deg = phi_lh * 180/np.pi
            calc_f = self.travel_time(startpoint=point, print_time='on', direction = 'forward')
            print('point after calc f', point)
            point[0] = point[0]/Rj
            point[2] = 2*np.pi - point[2]
            print('amended point', point)
            #print('got here, calc_f 0 =', calc_f[0], ' point = ' ,point)
            calc_b = self.travel_time(startpoint=point, print_time='on', direction = 'backward')
            time_f = calc_f[0]
            time_b = calc_b[0]
            time = time_b + time_f
            angle_time_dictionary[phi_lh] = time
        return angle_time_dictionary        

    def plot_angle_vs_time_both_directions(self, num = 50, r = 10):
        ''' generate a plot of how the travel time depends with the angle of the starting point. ''' 
        angles_times = self.multiple_travel_times_both_directions(num=num, r=r)
        #print(angles_times)
        angles = list(angles_times.keys())
        times = list(angles_times.values())
        angles_degree = [x*180/np.pi for x in angles]
        times_mins = [x/60 for x in times]
        #print(angles, times)
        fig, ax = plt.subplots()
        ax.plot(angles_degree, times_mins)
        ax.set(xlabel = 'phi $\u03BB_{III}$ (Degrees)', ylabel = 'Time (Minutes)', 
        title ='Effect of Starting longitude In Equatorial Plane on Travel Time along a field line \n That reaches r = {}$R_J$ in the equatorial plane'.format(r))
        #ax.tick_params(labelright = True)
        ax.grid()
        plt.show()

    def plot_multiple_distances_both_directions(self, num = 50, r = 10,):
        overall_distances = []
        phis = []
        startingPoints = []
        spacing = 2*np.pi/num
        for n in range(num):
            startingPoints.append([r, np.pi/2, n*spacing])
            phis.append(n*spacing)
        for point in startingPoints:
            print('New Startpoint = ', point)
            totaldistance = 0
            calc_f = self.travel_time(startpoint=point, print_time='on', direction = 'forward')
            print('point after calc f', point)
            point[0] = point[0]/Rj
            point[2] = 2*np.pi - point[2]
            print('amended point', point)
            #print('got here, calc_f 0 =', calc_f[0], ' point = ' ,point)
            calc_b = self.travel_time(startpoint=point, print_time='on', direction = 'backward')
            path_taken_f = calc_f[5]
            path_taken_b = calc_b[5]
            for i in range(len(path_taken_f)-1):
                start_point = path_taken_f[i]
                end_point = path_taken_f[i+1]
                difference = end_point - start_point
                distance = np.linalg.norm(difference)
                totaldistance += distance
            for i in range(len(path_taken_b)-1):
                start_point = path_taken_b[i]
                end_point = path_taken_b[i+1]
                difference = end_point - start_point
                distance = np.linalg.norm(difference)
                totaldistance += distance
            overall_distances.append(totaldistance)
        phis_degrees = np.array(phis) * 180/np.pi
        overall_distances_km = np.array(overall_distances) / 1e3
        fig, ax = plt.subplots()
        ax.plot(phis_degrees, overall_distances_km)
        ax.set(xlabel = 'phi $\u03BB_{III}$ (Degrees)', ylabel = 'Distance (Km)', 
        title ='Effect of Starting longitude In Equatorial Plane on Distance Travelled By Alfven Waves \n Along A field line passes through r = {}Rj in equatorial plane'.format(r))
        ax.grid(which = 'both')
        plt.show()

    def plot_radial_outflow_contour(self, mdot, gridsize = 40):
        gridx, gridy = self.makegrid_2d_negatives(200 ,gridsize= gridsize)
        vOutflows = []
        for i in range(len(gridy)):
            print('new row, {} to go'.format(len(gridx)-i))
            vOutflows_row = [] 
            for j in range(len(gridx)):
                x = gridx[i][j]
                y = gridy[i][j]
                r = np.sqrt(x**2 + y**2)
                if r < 6:
                    flow_vel = 10e3
                    vOutflows_row.append(flow_vel)
 
                    continue
                flow_vel = self.radial_flow_velocity(r, mdot)
                #if r < 0:
                
                vOutflows_row.append(flow_vel)
            vOutflows.append(vOutflows_row)
        
        outflows_km= np.array(vOutflows)/1e3

        fig, ax = plt.subplots(figsize = (25,16))
        lev_exp = np.arange(np.floor(np.log10(outflows_km.min())-1), np.ceil(np.log10(outflows_km.max())+1), step = 0.25)
        levs = np.power(10, lev_exp)
        cont = ax.contourf(gridx, gridy, outflows_km, cmap = 'cool', levels = levs, norm=mcolors.LogNorm())# locator=ticker.LogLocator())
        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        ax.add_patch(Circle((0,0), 6.2, color='c', zorder=90, label = "Io Orbital Radius"))
        ax.set_xlim(-gridsize,gridsize)
        ax.set_ylim(-gridsize,gridsize)
        ax.set(xlabel = '$R_J$ \n', ylabel = '$R_J$', title = 'Radial Outflow in Equatorial Plane \n mdot = {}'.format(mdot))
        fig.colorbar(cont, label = ' Radial Velocity $(kms^{-1})$')
        ax.set_aspect('equal', adjustable = 'box')
        ax.legend()
        ###plt.savefig('images-24-jan-update/v outflow equatorial')
        plt.show() 
        return vOutflows

    def plot_outflow_vs_alfven_eq_plane(self, mdot, gridsize = 20, model = 'VIP4', cansheet = 'off'):
        ''' 
        this incorrerctly assumes that radial outflow is in the equatorial plane, which is not the case

        '''
        theta = np.pi/2
        gridx, gridy = self.makegrid_2d_negatives(200 ,gridsize= gridsize)
        vOutflows = []
        vas = []
        va_over_outflow = []
        
        for i in range(len(gridy)):
            print('new row, {} to go'.format(len(gridx)-i))
            vOutflows_row = [] 
            Vas_row = []
            va_over_outflow_row = []
            for j in range(len(gridx)):
                
                x = gridx[i][j]
                y = gridy[i][j]
                r = np.sqrt(x**2 + y**2)
                if r < 6:
                    flow_vel = 10e3
                    vOutflows_row.append(flow_vel)
                    va = 1e6
                    Vas_row.append(va)
                    divided = va_corrected/flow_vel
                    va_over_outflow_row.append(divided)
                    continue
                flow_vel = self.flow_velocity(r, mdot)
                phi = np.arctan2(y,x) 

                n = self.density_comdefined(r, theta, 2*np.pi-phi)
                
                B_overall = self.mag_field_at_point(r, theta, phi)
                B_x, B_y, B_z = self.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi)
                B = np.array([B_x, B_y, B_z])
                B = B/(1e9) #CHRIS code outputs nT
                va = self.calculator(B, n)
                va_corrected = self.relativistic_correction(va)
                Vas_row.append(va_corrected)
                vOutflows_row.append(flow_vel)
                divided = va_corrected/flow_vel
                va_over_outflow_row.append(divided)
            vOutflows.append(vOutflows_row)
            vas.append(Vas_row)
            va_over_outflow.append(va_over_outflow_row)
             
        #outflows_km= np.array(vOutflows)/1e3
        #va_over_outflow = [va/vo for va,vo in zip(vas,vOutflows)]
        #va_over_outflow = []
        
        

        fig, ax = plt.subplots(figsize = (25,16))
        #lev_exp = np.arange(np.floor(np.log10(np.min(va_over_outflow))-1), np.ceil(np.log10(np.max(va_over_outflow))+1), step = 0.25)
        #levs = np.power(10, lev_exp)
        levs = [0,0.25,0.5,0.75,0.8,0.9,1,1.1,1.2,1.25,1.50,1.75,2,5,10]#, np.max(va_over_outflow)]
        plt.gca().patch.set_color('.25')
        cont = ax.contourf(gridx, gridy, va_over_outflow, cmap = 'seismic',norm = TwoSlopeNorm(vcenter = 1), levels = levs)#levels = levs, # norm=mcolors.LogNorm())# locator=ticker.LogLocator())
        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        ax.add_patch(Circle((0,0), 6.2, color='c', zorder=90, label = "Io Orbital Radius"))
        ax.set_xlim(-gridsize,gridsize)
        ax.set_ylim(-gridsize,gridsize)
        for r in np.arange(0, 115, 5):
            ax.add_patch(Circle((0,0), r, fill = False, color = 'lightgreen'))
        ax.set(xlabel = 'X $(R_J)$ \n', ylabel = 'Y $(R_J)$', title = 'Radial Outflow vs Alfven Velocity in Equatorial Plane \n mdot = {},'.format(mdot))
        fig.colorbar(cont, label = ' Alfven Velocity / Radial Velocity', ticks = levs)
        ax.set_aspect('equal', adjustable = 'box')
        ax.legend()
        ###plt.savefig('images-24-jan-update/v outflow vs VA equatorial')
        plt.show()

    def diverge_rel_correction(self, startpoint = [10, np.pi/2, 200.8* np.pi/180], direction = 'forward', rtol = 0.01):
        calc = self.travel_time(startpoint=startpoint, direction=direction)
        points = calc[5]
        corrected = calc[1]
        uncorrected = calc[2]
        diverge_index = self.find_index_diverge(corrected, uncorrected, rtol=rtol)
        
        diverge_point = points[diverge_index]
        diverge_point_sph = self.cart_to_sph(diverge_point[0], diverge_point[1], diverge_point[2])
        diverge_r = diverge_point_sph[0]/Rj
        diverge_colatitude = diverge_point_sph[1] * 180/np.pi
        print(diverge_r)
        return diverge_point, points, diverge_index
    def find_index_diverge(self, corrected, uncorrected, rtol):
            for i in range(len(corrected)):
                if not np.isclose(corrected[i], uncorrected[i], rtol=rtol):
                    print(i)
                    return i 

    
    def visualise_rel_correction_single_point(self, startpoint = [10, np.pi/2, 200.8* np.pi/180], rtol = 0.01):
        calc = self.diverge_rel_correction(startpoint=startpoint, rtol = rtol)
        diverge_point = calc[0]
        points = calc[1]
        diverge_index = calc[2]
        
        fig, ax = plt.subplots()
        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        ax.add_patch(Circle((0,0), 6.2, color='c', zorder=90, label = "Io Orbital Radius", fill = False))
        pre_diverge_points = points[:diverge_index]
        post_diverge_points = points[diverge_index:]
        pre_plottable = np.transpose(pre_diverge_points)
        pre_plottable_rj = pre_plottable/Rj
        pre_xs = pre_plottable_rj[0]
        pre_ys = pre_plottable_rj[1]
        pre_zs = pre_plottable_rj[2]
        print(len(pre_xs), len(pre_ys), len(pre_zs))
        pre_ss = []
        post_plottable = np.transpose(post_diverge_points)
        post_plottable_rj = post_plottable/Rj
        post_xs = post_plottable_rj[0]
        post_ys = post_plottable_rj[1]
        post_zs = post_plottable_rj[2]
        post_ss = []
        for i in range(len(pre_xs)):
            #phi = np.arctan2(ys[i],xs[i]) 
            pre_ss.append(np.sqrt(pre_xs[i]**2 + pre_ys[i]**2)) #* np.cos(phi - phi_lh_rad )) #np.cos(phi - phi_rh_rad )
        for i in range(len(post_xs)):
            #phi = np.arctan2(ys[i],xs[i]) 
            post_ss.append(np.sqrt(post_xs[i]**2 + post_ys[i]**2)) #* np.cos(phi - phi_lh_rad )) #np.cos(phi - phi_rh_rad )
        
        ax.plot(pre_ss, pre_zs, label = 'Field Line Before Correction Matters', Color = 'm')
        ax.plot(post_ss, post_zs, label = 'Field Line When Correction Matters', Color = 'b')
        ax.set_title('Path of Field Line Showing Where the relativisitc effect becomes important \n Tolerance = {}%'.format(rtol * 100))
        ax.set_aspect('equal', adjustable = 'box')
        ax.legend()
        plt.show()
    
    def relativistic_correction_area_of_impact_2d(self, phi_lh_deg ,rtol = 0.01, numpoints = 200):
        ''' 
        plot the areas in which the relativistic correction has an impact on the alfven velocity
        Inputs:
        phi_lh_deg - the longitude you want to view in left handed sysIII, in degrees
        rtol = the relative tolerance between uncorrected and corrected alfven velocity to define where the correction has an impact, default 1%
        '''
        
        phi_lh_rad = phi_lh_deg * np.pi/180
        
        ''' first define grid of points at which the velocities will be calculated ''' 
        grids, gridz = self.makegrid_2d_negatives(numpoints ,gridsize= 30)
        
        
        va_uncorrected = []
        va_corrected_list = []
         
        ''' the two for loops below calculate the uncorrected and corrected alfven velocity for every point in the grid ''' 
        for i in range(len(gridz)):
            print('new row, {} to go'.format(len(gridz)-i))
            va_uncorrected_row = []
            va_corrected_row = []
            firsttime = 0 #this is just used so that we only have to work out density at r = 6 once.

            for j in range(len(grids)):
                
                ''' for each point in the grid, work out the co-ordinates in spherical polar ''' 

                z = gridz[i][j]
                s = grids[i][j]

                r = np.sqrt(z**2 + s**2)
                phi = phi_lh_rad
                theta = np.arctan2(s,z)


                ''' things are a bit more awkward in the r<6 range and so is calculated seperately '''
                ''' first calculate density, then magnetic field, then alfven velocity, then corrected alfven velocity ''' 
                if r < 6: 
                    
                    n_at_6 = self.density_combined(6, theta, phi_lh_rad)
                    n = self.density_within_6(r, theta, phi_lh_rad, n_at_6) #in order to change the density profile within 6rj, this is what should be changed. 
                    ''' use chris' mag field models code to add the magnetic field from the internally generated field and the current sheet ''' 
                    B_overall = self.mag_field_at_point(r, theta, phi)
                    B_x, B_y, B_z = self.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi)
                    B = np.array([B_x, B_y, B_z])
                    B_tesla = B/(1e9) #CHRIS code outputs nT
                    va = self.calculator(B_tesla, n)
                    va_uncorrected_row.append(va)
                    va_corrected = self.relativistic_correction(va)
                    va_corrected_row.append(va_corrected)
                    continue #after its calculated for r<6 back to the top of the loop
            
                ''' if r>6, then density and mangetic field calculated in the same manner ''' 
                n = self.density_combined(r, theta, phi_lh_rad)
                #print(n)
                B_overall = self.mag_field_at_point(r, theta, phi)
                B_x, B_y, B_z = self.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi)
                B = np.array([B_x, B_y, B_z])
                B = np.array(B)
                B_tesla = B/(1e9) #CHRIS code outputs nT
                va = self.calculator(B_tesla, n)
                va_uncorrected_row.append(va)
                corrected_va = self.relativistic_correction(va)
                va_corrected_row.append(corrected_va)
            va_uncorrected.append(va_uncorrected_row)
            va_corrected_list.append(va_corrected_row)

        ''' np.isclose() works with array_like objects. will return a list similar to [[True, True, True, False...]]. hopefully works with nested arrays? '''
        ''' then turn all Trues into 1, all False into 0 ''' 

        are_close = np.isclose(va_uncorrected, va_corrected, rtol = rtol)    
        ''' this will turn the false and trues into 0s and 1s ''' 
        are_close = are_close*1

        ''' plot ''' 
        '''
        fig, ax = plt.subplots()
        plt.gca().patch.set_color('.25')
        #print(are_close)

        cont = ax.contourf(grids, gridz, are_close, cmap = 'hot',levels = [0.2,0.9])#levels = levs, # norm=mcolors.LogNorm())# locator=ticker.LogLocator())

        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        ax.add_patch(Circle((0,0), 6.2, color='c', zorder=90, label = "Io Orbital Radius", fill = False))
        ax.set_xlim(-30,30)
        ax.set_ylim(-30,30)
        for r in np.arange(0, 45, 5):
            ax.add_patch(Circle((0,0), r, fill = False, color = 'lightgreen'))
        ax.set(xlabel = 'X $(R_J)$ \n', ylabel = 'Y $(R_J)$', title = 'Where does Correction for alfven velocity matter \n  equators = {}, rtol = {}'.format(equators, rtol))
        fig.colorbar(cont) 
        
        ax.legend()
        ###plt.savefig('images-24-jan-update/where va correction matterss 2d')
        plt.show()
        '''
        ''' plot v2 ''' 
        divided = np.array(va_corrected_list)/np.array(va_uncorrected)
        fig, ax = plt.subplots()
        plt.gca().patch.set_color('.25')
        #print(are_close)

        cont = ax.contourf(grids, gridz, divided, levels = [0,0.5,0.6,0.7,0.8,0.9,0.95,0.96,0.97,0.98,0.99,1], colors = personal_cmap)#levels = [0.2,0.9])#levels = levs, # norm=mcolors.LogNorm())# locator=ticker.LogLocator())

        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        ax.add_patch(Circle((0,0), 6.2, color='k', zorder=90, label = "Io Orbital Radius", fill = False))
        ax.set_xlim(-30,30)
        ax.set_ylim(-30,30)
        #for r in np.arange(0, 45, 5):
         #   ax.add_patch(Circle((0,0), r, fill = False, color = 'lightgreen'))
        ax.set(xlabel = ' $(R_J)$ \n', ylabel = ' $(R_J)$', title = 'Corrected/Uncorrected Alfven velocity in phi = {:.0f} plane \n '.format(phi_lh_deg, rtol))
        fig.colorbar(cont) 
        ax.set_aspect('equal', adjustable = 'box')
        ax.legend()
        ###plt.savefig('images-24-jan-update/where va correction matters 2d v2')
        plt.show()

    def relativistic_correction_area_of_impact_eq_plane(self):
        ''' 

        '''
        theta = np.pi/2
        gridx, gridy = self.makegrid_2d_negatives(200 ,gridsize= 30)
        va_uncorrected_list = []
        va_corrected_list = []
        va_corrected_over_uncorrected = []
        
        for i in range(len(gridy)):
            print('new row, {} to go'.format(len(gridx)-i))
            va_uncorrected_row = []
            va_corrected_row = []
            va_corrected_over_uncorrected_row = []
            firsttime = 0 #this is just used so that we only have to work out density at r = 6 once.
            for j in range(len(gridx)):
                x = gridx[i][j]
                y = gridy[i][j]
                r = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y,x)
                if r < 6:

                    n_at_6 = self.density_combined(r, theta, phi)
                    
                    n = self.density_within_6(r, theta, phi, n_at_6) #in order to change the density profile within 6rj, this is what should be changed. 
                    ''' use chris' mag field models code to add the magnetic field from the internally generated field and the current sheet ''' 
                    B_overall = self.mag_field_at_point(r, theta, phi)
                    B_x, B_y, B_z = self.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi_rh)
                    B = np.array([B_x, B_y, B_z])
                    B = np.array(B)
                    B_tesla = B/(1e9) #CHRIS code outputs nT
                    va = self.calculator(B_tesla, n)
                    va_uncorrected_row.append(va)
                    va_corrected = self.relativistic_correction(va)
                    va_corrected_row.append(va_corrected)
                    divided = va_corrected/va
                    va_corrected_over_uncorrected_row.append(divided)
                    continue

                n = self.density_combined(r, theta, phi_lh)
                #print(n)

                B_overall = self.mag_field_at_point(r, theta, phi)
                B_x, B_y, B_z = self.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi_rh)
                B = np.array([B_x, B_y, B_z])
                B = np.array(B)
                B_tesla = B/(1e9) #CHRIS code outputs nT
                va = self.calculator(B_tesla, n)
                va_uncorrected_row.append(va)
                corrected_va = self.relativistic_correction(va)
                va_corrected_row.append(corrected_va)
                divided = va_corrected/va
                va_corrected_over_uncorrected_row.append(divided)
            va_corrected_list.append(va_corrected_row)
            va_uncorrected_list.append(va_uncorrected_row)
            va_corrected_over_uncorrected.append(va_corrected_over_uncorrected_row)
        
        fig, ax = plt.subplots()
        plt.gca().patch.set_color('.25')

        cont = ax.contourf(gridx, gridy, va_corrected_over_uncorrected, levels = [0,0.5,0.6,0.7,0.8,0.9,0.95,0.96,0.97,0.98,0.99,1], colors = personal_cmap)#levels = [0.2,0.9])#levels = levs, # norm=mcolors.LogNorm())# locator=ticker.LogLocator())

        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        ax.add_patch(Circle((0,0), 6.2, color='k', zorder=90, label = "Io Orbital Radius", fill = False))
        ax.set_xlim(-30,30)
        ax.set_ylim(-30,30)
        for r in np.arange(0, 45, 5):
            ax.add_patch(Circle((0,0), r, fill = False, color = 'firebrick'))
        ax.set(xlabel = ' $(R_J)$ \n', ylabel = ' $(R_J)$', title = 'Corrected/Uncorrected Alfven velocity in equatorial plane \n  ')
        fig.colorbar(cont) 
        ax.set_aspect('equal', adjustable = 'box')
        ax.legend()
        ###plt.savefig('images-24-jan-update/where va correction matters 2d topdown')
        plt.show()

            
        #outflows_km= np.array(vOutflows)/1e3
        #va_over_outflow = [va/vo for va,vo in zip(vas,vOutflows)]
        #va_over_outflow = []
        
        

        fig, ax = plt.subplots(figsize = (25,16))
        #lev_exp = np.arange(np.floor(np.log10(np.min(va_over_outflow))-1), np.ceil(np.log10(np.max(va_over_outflow))+1), step = 0.25)
        #levs = np.power(10, lev_exp)
        


    def difference_in_travel_time(self, r, phi_lh_rad):
        theta = np.pi/2
        point = [r, theta, phi_lh_rad]
        phis = []
        calc_f = self.travel_time(startpoint=point, print_time='on', direction = 'forward')
        point[0] = point[0]/Rj
        point[2] = 2*np.pi - point[2]
        calc_b = self.travel_time(startpoint=point, print_time='on', direction = 'backward')
        time_f = calc_f[0]
        time_b = calc_b[0]
        difference = abs(time_f - time_b)
        return difference 
    def difference_in_tt_multi(self, r = 8, num = 10):
        startingPoints = []
        differences = []
        phis = []
        spacing = 2*np.pi/num
        for n in range(num):
            startingPoints.append([r, np.pi/2, n*spacing])
            phis.append(n*spacing*180/np.pi)
        for point in startingPoints:
            
            
            phi_lh = point[2]
            if self.aligned == 'yes':
                   colatitude = np.pi/2
            if self.aligned == 'no': 
                cent_eq_latitude = self.centrifugal_equator(r, 2*np.pi - phi_lh)
                colatitude = np.pi/2 - cent_eq_latitude
            point[1] = colatitude
            print('New Startpoint, ', point)
            phi_lh_deg = phi_lh * 180/np.pi
            calc_f = self.travel_time(startpoint=point, print_time='off', direction = 'forward')
            point[0] = point[0]/Rj
            point[2] = 2*np.pi - point[2]
            point[1] = colatitude
            print('amended point', point)
            #print('got here, calc_f 0 =', calc_f[0], ' point = ' ,point)
            calc_b = self.travel_time(startpoint=point, print_time='off', direction = 'backward')
            time_f = calc_f[0]
            time_b = calc_b[0]
            difference = abs(time_b - time_f)
            differences.append(difference)
        fig, ax = plt.subplots()
        ax.plot(phis, differences, color = 'k')
        #ax.grid()
        ax.set( xlabel = '$\u03BB_{III}$ (Degrees)',
         ylabel = 'Difference in Time (Secconds)') #title = 'Difference of Travel time to north Vs South hemisphere \n Dependence on longitude'
        plt.show()

    def db6_better(self, phi_lh_deg, numpoints = 200, mdots = [500,1300,2000], stop = 60, corotation = False, azimuthal = False, show = True):
        ''' for a given longitude, calculate radial outflow velocity vs local alfven velocity'''
        ''' first, all of the outflow velocities for all of the given mdots '''
        rs = np.linspace(6,stop,numpoints)
        #mdots = [500,1300,2000]
        mdot_outflows = {}
        for mdot in mdots:
            outflow_for_given_mdot = []
            for r in rs:
                outflow = self.radial_flow_velocity(r, mdot)
                outflow_for_given_mdot.append(outflow)
            mdot_outflows[mdot] = outflow_for_given_mdot


        ''' next, we need to calc alfven velocity, at each of the r values, along centrifigul equator, for given phi '''
        vas = []
        phi_rh_rad = (360 -phi_lh_deg) *np.pi/180
        phi_lh_rad = phi_lh_deg * np.pi/180
        corotations = []
        for r in rs:
            
            if self.aligned == 'no':
                cent_eq_latitude = self.centrifugal_equator(r, phi_rh_rad)
                colatitude = np.pi/2 - cent_eq_latitude
            else:
                colatitude = np.pi/2
            n = self.density_combined(r, colatitude, phi_lh_rad)
            B_overall = self.mag_field_at_point(r, colatitude, phi_rh_rad)
            B_x, B_y, B_z = self.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, colatitude, phi_rh_rad)
            B = np.array([B_x, B_y, B_z])
            B =  B/(10**9)  #chris's code is in nT
            va = self.calculator(B, n)
            vas.append(va)
            if corotation:
                corotations.append((2*np.pi/(9.93*3600))* r * Rj)

        

        ''' and then we plot ''' 
        if show: 
            fig, ax = plt.subplots()
        if azimuthal:
            azimuthals = []
            ang_vel = np.load('angular_velocity_data/mdot_2000.0.npy', allow_pickle=True)
            ang_vel_rs = np.load('angular_velocity_data/radii2000.0.npy',  allow_pickle=True)
            for i in range(len(ang_vel_rs)):
                r = ang_vel_rs[i]
                omega = ang_vel[i]
                azimuthals.append((omega_J-omega*omega_J)*r*Rj)
                #zimuthals.append((omega*omega_J)*r*Rj)
            #print(azimuthals)
            azimuthals_km = np.array(azimuthals)/1e3
            if show:
                ax.plot(ang_vel_rs, azimuthals_km, label = 'Azimuthal Velocity (Dipole Approx. {} = 2000kg/s)'.format(u'\u1E41'))
        if show:
            for key in mdot_outflows:
                v_kms = np.array(mdot_outflows[key]) /1000
                ax.plot(rs, v_kms, label = 'Radial Velocity ({} = {}Kg/s)'.format(u'\u1E41' ,key))
            vas_km = np.array(vas)/1000
            ax.plot(rs, vas_km, label = 'Local Alfven Velocity')
            if corotation:
                corotations_km = np.array(corotations)/1e3
                ax.plot(rs, corotations_km, label = 'Corotation Speed', color = 'm')
        #ax.grid()
            ax.legend()
            ax.yaxis.set_ticks_position('both')
            plt.yscale("log")
            ax.set_ylim(1,1000)
            ax.set_xlim(0,100)
            #ax.set(title = 'Radial Outflow Along Centrifugal Equator And Local Alfven Velocity \n For $\u03BB_{{III}} $ Longitude of {:.1f}{} '.format(phi_lh_deg, u"\N{DEGREE SIGN}"),
            ax.set(xlabel = 'R ($R_J$)',ylabel = 'Velocity ($kms^{-1}$)')
            ###plt.savefig('images-24-jan-update/radial_flow_plot_better.png'):
            plt.show()
        if azimuthal:
            return mdot_outflows, vas, rs, azimuthals, ang_vel_rs
        return mdot_outflows, vas, rs
    
    def find_va_vo_cross_point(self, mdots, phi_lh_deg, numpoints = 200):
        mdot_crossing = {}
        mdot_outflows, vas, rs= self.db6_better(phi_lh_deg=phi_lh_deg, mdots=mdots, stop = 80, show = False, numpoints=numpoints)
        for key in mdot_outflows:
            outflows = mdot_outflows[key]
            for i in range(len(outflows)):
                if outflows[i] > vas[i]:
                    mdot_crossing[key] = rs[i]
                    break
        return mdot_crossing

    def find_va_vphi_cross_point_lray(self, phi_lh_deg):
        results = self.db6_better(phi_lh_deg=phi_lh_deg, mdots=[2000], stop = 80, azimuthal=True, show= False, numpoints=400)
        vas = results[1]
        vas_rs = results[2]
        azis = results[3]
        azis_rs = results[4]
        for i in range(len(vas)):
            va = vas[i]
            r = vas_rs[i]
            try:
                index_point = list(azis_rs).index(r)
                azi = azis[index_point]
            except ValueError:
                #print('rs dont match up and i am sad about it')
                finding_index = [np.abs(x-r) for x in azis_rs]
                closest_r_index = np.array(finding_index).argmin()
                azi = azis[closest_r_index]
            
            if azi > va:
                return r
            

    def find_va_vphi_cross_point_pensionerov(self, phi_lh_deg):
        results = self.db6_better(phi_lh_deg=phi_lh_deg, mdots=[2000], stop = 80, azimuthal=True, show= False, numpoints=400)
        vas = results[1]
        vas_rs = results[2]
        lt = self.what_LT_segment(phi_lh_deg=phi_lh_deg)
        with open('angular_velocity_data/Pensionerov_et_al/{}.txt'.format(lt), 'r') as f:
            data = f.read().splitlines()
        data = np.array([data[i].split() for i in range(len(data))])
        data = data.flatten()
        data = [data[i].replace(',', '') for i in range(len(data))]
        data = [float(data[i]) for i in range(len(data))]
        r_ang_vel = [data[i:i+2] for i in range(0, len(data), 2)]
        rs_vels = np.transpose(r_ang_vel)

        azis = []
        azis_rs = []
        for i in range(len(rs_vels[0])):
            r = rs_vels[0][i]
            omega = rs_vels[1][i]
            azis.append(r*Rj*omega*omega_J)
            azis.append((omega_J-omega*omega_J)*r*Rj)
            #print(r, omega, omega_J, r*Rj*omega*omega_J )
            azis_rs.append(r)
        for i in range(len(vas)):
            va = vas[i]
            r = vas_rs[i]
            try:
                index_point = list(azis_rs).index(r)
                azi = azis[index_point]
            except ValueError:
                #print('rs dont match up and i am sad about it')
                finding_index = [np.abs(x-r) for x in azis_rs]
                closest_r_index = np.array(finding_index).argmin()
                azi = azis[closest_r_index]
            
            if azi > va:
                return r


    def outflow_vs_alfven_cent_plane(self, mdot, gridsize = 60):
        '''  
        '''
        theta = np.pi/2
        gridx, gridy = self.makegrid_2d_negatives(200 ,gridsize= gridsize)
        vOutflows = []
        vas = []
        va_over_outflow = []
        
        for i in range(len(gridy)):
            print('new row, {} to go'.format(len(gridx)-i))
            vOutflows_row = [] 
            Vas_row = []
            va_over_outflow_row = []
            for j in range(len(gridx)):
                
                x = gridx[i][j]
                y = gridy[i][j]
                r = np.sqrt(x**2 + y**2)
                if r < 6:
                    flow_vel = 10e3
                    vOutflows_row.append(flow_vel)
                    va = 1e6
                    Vas_row.append(va)
                    divided = va_corrected/flow_vel
                    va_over_outflow_row.append(divided)
                    continue
                flow_vel = self.radial_flow_velocity(r, mdot)
                phi = np.arctan2(y,x) 
                if self.aligned == 'yes':
                   colatitude = np.pi/2
                if self.aligned == 'no': 
                    cent_eq_latitude = self.centrifugal_equator(r, phi)
                    colatitude = np.pi/2 - cent_eq_latitude
                n = self.density_combined(r, colatitude, 2*np.pi-phi)
                B_overall = self.mag_field_at_point(r, colatitude, phi)
                B_x, B_y, B_z = self.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, colatitude, phi)
                B = np.array([B_x, B_y, B_z])
                B = B/(1e9) #CHRIS code outputs nT
                va = self.calculator(B, n)
                va_corrected = self.relativistic_correction(va)
                Vas_row.append(va_corrected)
                vOutflows_row.append(flow_vel)
                divided = va_corrected/flow_vel
                va_over_outflow_row.append(divided)
            vOutflows.append(vOutflows_row)
            vas.append(Vas_row)
            va_over_outflow.append(va_over_outflow_row)
             
        #outflows_km= np.array(vOutflows)/1e3
        #va_over_outflow = [va/vo for va,vo in zip(vas,vOutflows)]
        #va_over_outflow = []
        
        

        fig, ax = plt.subplots(figsize = (25,16))
        #lev_exp = np.arange(np.floor(np.log10(np.min(va_over_outflow))-1), np.ceil(np.log10(np.max(va_over_outflow))+1), step = 0.25)
        #levs = np.power(10, lev_exp)
        levs = [0,0.25,0.5,0.75,0.8,0.9,1,1.1,1.2,1.25,1.50,1.75,2,5,10]#, np.max(va_over_outflow)]
        plt.gca().patch.set_color('.25')
        cont = ax.contourf(gridx, gridy, va_over_outflow, cmap = 'seismic',norm = TwoSlopeNorm(vcenter = 1), levels = levs)#levels = levs, # norm=mcolors.LogNorm())# locator=ticker.LogLocator())
        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        ax.add_patch(Circle((0,0), 6.2, color='c', zorder=2, label = "Io Orbital Radius"))
        ax.set_xlim(-gridsize,gridsize)
        ax.set_ylim(-gridsize,gridsize)
        for r in np.arange(0, 115, 5):
            ax.add_patch(Circle((0,0), r, fill = False, color = 'mediumvioletred', zorder = 5))
        ax.set(xlabel = 'X $(R_J)$ \n', ylabel = 'Y $(R_J)$')#, title = 'Radial Outflow vs Alfven Velocity in Centtrifugal Plane \n mdot = {} \n {}'.format(mdot, self.plot_label))
        fig.colorbar(cont, label = ' Alfven Velocity / Radial Velocity', ticks = levs)
        ax.set_aspect('equal', adjustable = 'box')
        ax.grid(False)
        ax.legend()
        ###plt.savefig('images-24-jan-update/v outflow vs VA equatorial')
        plt.show()

    def lat_where_va_correction_matters(self, r, phi_lh_deg, step = 1*np.pi/180, rtol = 0.01):
        '''
        input r in rj, phi sys III left handed 
        return latitude where the relativistic correction matters
        '''
        '''
        it'll probably be faster starting at vertically up and working out where it stops to matter?
        '''
        phi_lh_rad = phi_lh_deg * np.pi / 180
        phi_rh_rad = 2*np.pi - phi_lh_rad
        thetas = np.arange(2*np.pi/180, np.pi,step = step)
        firsttime = 0
        #print(thetas)
        for theta in thetas:
            n = self.density_combined(r, theta, phi_lh_rad)
            #print(n)

            B_overall = self.mag_field_at_point(r, theta, phi_rh_rad)
            B_x, B_y, B_z = self.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi_rh_rad)
            
            B = np.array([B_x, B_y, B_z])
            B_tesla = B/(1e9) #CHRIS code outputs nT
            va = self.calculator(B_tesla, n)
            corrected_va = self.relativistic_correction(va)
            if np.isclose(corrected_va, va, rtol = rtol):
                #print(theta)
                if firsttime ==0:
                    value_1 = theta
                    firsttime = 1
                value_2 = theta
                #print(value)
        index_t = np.where(thetas == value_1)[0]
        index_b = np.where(thetas == value_2)[0]
        
        first_theta_it_matters_colat = thetas[index_t[0]-1]
        second_timc = thetas[index_b[0]-1]
        
        lat_top = np.pi/2 - first_theta_it_matters_colat
        lat_deg_top = lat_top * 180/np.pi

        lat_bottom = np.pi/2 - second_timc
        lat_deg_bottom = lat_bottom * 180/np.pi
        

        return lat_deg_top, lat_deg_bottom

    def rel_correction_latitude_contour(self, rtol = 0.01, hemisphere = 'n', num=50):
        ''' '''
        rs = np.linspace(6,60, num = num)
        phis = np.linspace(0,360, num = num)
        thetas_where_matters = []
        for r in rs:
            thetas_row = []
            for phi in phis:
                if hemisphere == 'n':
                    theta = self.lat_where_va_correction_matters(r, phi, rtol =rtol)[0]
                    destination_label = 'northern hemisphere'
                if hemisphere == 's':
                   theta = self.lat_where_va_correction_matters(r, phi= rtol)[1] 
                   destination_label = 'southern hemisphere'
                thetas_row.append(theta)
            thetas_where_matters.append(thetas_row)
        fig, ax = plt.subplots()
        thetas_where_matters = np.clip(thetas_where_matters,0,90)
        #levs = np.linspace(np.amin(thetas_where_matters), np.amax(thetas_where_matters), 30)
        levs = np.arange(10,90 ,5)
        cont = ax.contourf(phis,rs,thetas_where_matters, cmap = 'bone', levels = levs)
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        
        ax.set(ylabel = 'R $(R_J)$', xlabel = 'Longitude $\u03BB_{{III}}$')#), title = 'tolerance = {} \n {} \n {}'.format(rtol,self.plot_label, destination_label))
        fig.colorbar(cont, label = 'Latitude', ticks = levs)
        plt.show()
    def va_along_field_line(self, startpoint = [10, np.pi/2, 200.8*np.pi/180], breakpoint= 6, direction = 'forward', uncorrected = 'off'):
        startpoint[0] = startpoint[0]*Rj
        phi_lh = startpoint[2]
        field_line = self.trace_magnetic_field(starting_cordinates=startpoint, one_way='on', break_point=breakpoint, step = 0.001, pathing= direction)
        Bs = np.array(field_line[2]) / 1e9
        points = np.array(field_line[0])
        vas_corrected = []
        vas_uncorrected = []
        ns = []
        rs = []
        ss =[0]
        latitudes = []
        latitudes_deg = []
        for point in points:
            x,y,z = point[0], point[1], point[2]
            r, theta, phi_rh = self.cart_to_sph(x,y,z)
            r = r/Rj
            latitude = np.pi/2 - theta
            latitude_deg = latitude * 180/np.pi
            latitudes_deg.append(latitude_deg)
            phi_lh = 2*np.pi - phi_rh
            n = self.density_combined(r, theta, phi_lh)
            ns.append(n)
            rs.append(r)
        for i in range(len(ns)):
            n = ns[i]
            B = Bs[i]
            va_uncorrected = self.calculator(B, n)
            va_corrected = self.relativistic_correction(va_uncorrected)
            vas_uncorrected.append(va_uncorrected)
            vas_corrected.append(va_corrected)
        sumtotal = 0
        for i in range(len(rs)-1):
            absDifference = abs(rs[i+1] - rs[i])
            sumtotal += absDifference
            ss.append(sumtotal)
        fig, ax = plt.subplots()
        ax.plot(ss, vas_corrected, label = '$V_A$ inc. Relativistic Correction')
        if uncorrected == 'on':
            ax.plot(ss, vas_uncorrected, label = '$V_A$ not inc. Relativistic Correction')
        ax.set( xlabel = 'Distance Along Field Line $(R_J)$', ylabel = 'Alfven Velocity')
        #title = '{} \n startpoint = {}'.format(self.plot_label, startpoint)
        plt.show()
        return ss, vas_corrected, vas_uncorrected, rs, latitudes_deg

    def va_along_field_line_both_directions(self, startpoint = [10, np.pi/2, 200.8*np.pi/180], breakpoint= 2, direction = 'forward', uncorrected = 'off'):
        startpoint[0] = startpoint[0]*Rj
        #print(startpoint)
        phi_lh = startpoint[2]
        startpoint_copy = deepcopy(startpoint)
        #print(startpoint_copy)
        field_line_f = self.trace_magnetic_field(starting_cordinates=startpoint, one_way='on', break_point=breakpoint, step = 0.001, pathing= 'forward')
        #print(startpoint_copy)
        field_line_b = self.trace_magnetic_field(starting_cordinates=startpoint_copy, one_way='on', break_point=breakpoint, step = 0.001, pathing= 'backward')
        Bs_f = np.array(field_line_f[2]) / 1e9
        points_f = np.array(field_line_f[0])
        
        
        Bs_b = np.array(field_line_b[2]) / 1e9
        points_b = np.array(field_line_b[0])
        points_b = np.flip(points_b, axis = 0)
        Bs_b = np.flip(Bs_b, axis = 0)
        Bs = np.append(Bs_b, Bs_f, axis = 0)
        points = np.append(points_b, points_f, axis = 0)
        vas_corrected = []
        vas_uncorrected = []
        ns = []
        rs = []
        ss = [0]
        latitudes = []
        thetas = []
        latitudes_deg = []
        for point in points:
    
            x,y,z = point[0], point[1], point[2]
            r, theta, phi_rh = self.cart_to_sph(x,y,z)
            r = r/Rj
            thetas.append(theta)
            latitude = np.pi/2 - theta
            latitude_deg = latitude * 180/np.pi
            latitudes_deg.append(latitude_deg)
            latitudes.append(latitude)
            phi_lh = 2*np.pi - phi_rh
            n = self.density_combined(r, theta, phi_lh)
            ns.append(n)
            rs.append(r)
        for i in range(len(ns)):
            n = ns[i]
            B = Bs[i]
            va_uncorrected = self.calculator(B, n)
            va_corrected = self.relativistic_correction(va_uncorrected)
            vas_uncorrected.append(va_uncorrected)
            vas_corrected.append(va_corrected)
        sumtotal = 0
        for i in range(len(rs)-1):
            absDifference = abs(rs[i+1] - rs[i])
            sumtotal += absDifference
            ss.append(sumtotal)
        magBs = []
        for b in Bs:
            magBs.append(np.linalg.norm(b))
        #fig, ax = plt.subplots()
        #ax.plot(thetas, vas_corrected, label = '$V_A$ inc. Relativistic Correction')
        #plt.show()
    
        return ss, vas_corrected, vas_uncorrected, rs, latitudes_deg, latitudes, thetas, ns, magBs

    def bn_along_field_line(self ,startpoint = [10, np.pi/2, 200.8*np.pi/180], breakpoint= 2, limits = False, clipped = False, show = True):
        results = self.va_along_field_line_both_directions(startpoint, breakpoint)
        latitudes_deg = np.array(results[4])
        ns = np.array(results[7])/1e6
        Bs = np.array(results[8])*1e9
        if show:
            fig, ax1 = plt.subplots()
            ax1.plot(latitudes_deg, ns, color = 'k')#, label = 'Density $(cm')
            ax1.set(xlabel = 'Latitude (Degrees)', ylabel = 'Density (cm$^{-3}$)')
            plt.yscale('log')
            ax2 = ax1.twinx() 
            ax1.grid(None)
            ax2.plot(latitudes_deg, Bs, color = 'r')# label = 'r ($R_J$)', color = 'c', linestyle ='--')
            ax2.set_ylabel('Magnetic Field Strength (nT)', color = 'r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.grid(None)
            if limits:
                ax1.set_xlim(-40,40)
                ax2.set_xlim(-20,20)
                ax1.set_ylim(0,1e3)
                ax2.set_ylim(1e3,1e5)
            plt.yscale('log')
            plt.show()
        if clipped:
            clipped_bs = []
            clipped_ns = []
            clipped_latitude_degs = latitudes_deg[(latitudes_deg >= -40) & (latitudes_deg <= 40)]
            for lat in clipped_latitude_degs:
                index = np.where(latitudes_deg == lat)
                #print(index[0][0])
                clipped_bs.append(Bs[index[0][0]])
                clipped_ns.append(ns[[index][0][0]][0])
                print(ns[[index][0][0]][0])
            #print(clipped_ns)
            fig, ax1 = plt.subplots()
            #print(clipped_ns)
            ax1.plot(clipped_latitude_degs, clipped_ns, color = 'k')#, label = 'Density $(cm')
            ax1.set(xlabel = 'Latitude (Degrees)', ylabel = 'Density (cm$^{-3}$)')
            plt.yscale('log')
            ax2 = ax1.twinx() 
            ax1.grid(None)
            ax2.plot(clipped_latitude_degs, clipped_bs, color = 'r')# label = 'r ($R_J$)', color = 'c', linestyle ='--')
            ax2.set_ylabel('Magnetic Field Strength (nT)', color = 'r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.grid(None)
            plt.yscale('log')
            plt.show()
        

        
    def alfven_at_point(self, point):
        '''
        r in rj
        theta in colatitude (degree)
        phi in lh longitude (degree)
        '''
        dtor = np.pi/180
        r = point[0]
        theta = point[1] * dtor
        phi_lh = point[2] * dtor
        phi_rh = 2*np.pi - phi_lh
        n = self.density_combined(r,theta,phi_lh)
        B_overall = self.mag_field_at_point(r, theta, phi_rh)
        B_x, B_y, B_z = self.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi_rh)
        B = np.array([B_x, B_y, B_z])
        B = B/(1e9) #CHRIS code outputs nT
        va = self.calculator(B, n)
        va_corrected = self.relativistic_correction(va)
        return va_corrected


    def pensionerov_ang_vel_recreate(self, azimuthal = False):
        LT_segments = ['LT00', 'LT03', 'LT06', 'LT09', 'LT12' , 'LT15', 'LT18', 'LT21']
        lt_r_vel = {}
        for lt in LT_segments:
            print(lt)
            with open('angular_velocity_data/Pensionerov_et_al/{}.txt'.format(lt), 'r') as f:
                data = f.read().splitlines()
                data = np.array([data[i].split() for i in range(len(data))])
                data = data.flatten()
                data = [data[i].replace(',', '') for i in range(len(data))]
                data = [float(data[i]) for i in range(len(data))]
                r_ang_vel = [data[i:i+2] for i in range(0, len(data), 2)]
                lt_r_vel[lt] = r_ang_vel
        fig, ax = plt.subplots()
        colours = ['b','g','r','c','m','k','y', 'lime']
        color_index = 0
        for key in lt_r_vel:
            r_vel = lt_r_vel[key]
            #print(r_vel)
            rs_vels = np.transpose(r_vel)
            #print(rs_vels)
            smoothed = savgol_filter(rs_vels[1],71,7)
            smoother = savgol_filter(smoothed, 71,7)
            ax.plot(rs_vels[0], smoother, label = '{}'.format(key), color = colours[color_index])
            color_index+=1 
        ax.set_xlim(0,70)
        if azimuthal:
            azimuthals = []
            ang_vel = np.load('angular_velocity_data/mdot_2000.0.npy', allow_pickle=True)
            ang_vel_rs = np.load('angular_velocity_data/radii2000.0.npy',  allow_pickle=True)
            for i in range(len(ang_vel_rs)):
                r = ang_vel_rs[i]
                omega = ang_vel[i]
                azimuthals.append(r*Rj*omega*omega_J)
            #azimuthals_km = np.array(azimuthals)/1e3
            ax.plot(ang_vel_rs, ang_vel, label = 'Ray Model', color = "peru")
        ax.axvline(x = 21.2, label= 'Approximate location where $v_{\u03A6} > v_A$', color = 'firebrick', linestyle = '--')
        ax.axvline(x = 26, label = 'Approximate location where $v_r > v_A$', color = 'lightcoral', linestyle = '--')
        ax.legend()
        ax.set(ylabel = 'Angular Velocity ($\u03A9_J$)', xlabel = 'r ($R_J$)')
        plt.show()
        
    def what_LT_segment(self, phi_lh_deg= 0):
        LT = 0
        if 0<=phi_lh_deg<=45:
            LT = 'LT21'
        if 45<phi_lh_deg<=90:
            LT = 'LT18'
        if 90<phi_lh_deg<=135:
            LT = 'LT15'
        if 135<phi_lh_deg<=180:
            LT = 'LT12'
        if 180<phi_lh_deg<=225:
            LT = 'LT09'
        if 225<phi_lh_deg<=270:
            LT = 'LT06'
        if 270<phi_lh_deg<=315:
            LT = 'LT03'        
        if 315<phi_lh_deg<=360:
            LT = 'LT00'
        return LT

    def find_cross_points_lray(self, numpoints = 10):
        phis = np.linspace(0, 360, num = numpoints)
        phi_outflow_azi = {}
        for phi in phis:
            print('new phi, phi = {}'.format(phi))
            outflow_crossing = self.find_va_vo_cross_point([2000], phi_lh_deg=phi, numpoints=400)
            outflow_crossing_r = outflow_crossing[2000]
            azi_crossing_r = self.find_va_vphi_cross_point_lray(phi_lh_deg=phi)
            phi_outflow_azi[phi] = [outflow_crossing_r, azi_crossing_r]
        with open('crossing_data_ray_new.json', 'w') as fp:
            json.dump(phi_outflow_azi, fp, indent=4)

    def find_cross_points_pensionerov(self, numpoints = 10):
        phis = np.linspace(0, 360, num = numpoints)
        phi_outflow_azi = {}
        for phi in phis:
            print('new phi, phi = {}'.format(phi))
            outflow_crossing = self.find_va_vo_cross_point([2000], phi_lh_deg=phi, numpoints=400)
            outflow_crossing_r = outflow_crossing[2000]
            azi_crossing_r = self.find_va_vphi_cross_point_pensionerov(phi_lh_deg=phi)
            phi_outflow_azi[phi] = [outflow_crossing_r, azi_crossing_r]
        with open('crossing_data_pensionerov_new.json', 'w') as fp:
            json.dump(phi_outflow_azi, fp, indent=4)

    def analyse_cross_points(self, model = 'ray', winners = False, closest_to_planet = False, outflow_cross_plot = False, furthest_azi_cross = False, footprint = False, 
    show = False, phi_cross_plot = False, outflow_footprint = False, footprint_direction = 'south'):
        if model == 'ray':
            with open('crossing_data_ray_new.json') as json_file:
                data = json.load(json_file)
        if model == 'pensionerov':
            with open('crossing_data_pensionerov_new.json') as json_file:
                data = json.load(json_file)
        if model == 'test':
            with open('test.json') as json_file:
                data = json.load(json_file)
        phis = list(data.keys())
        if winners:
            outflow_winners = 0
            azi_winners = 0
            for phi in phis:
                outflow_cross = data[phi][0]
                azi_cross = data[phi][1]
                if outflow_cross < azi_cross:
                    outflow_winners +=1 
                else:
                    azi_winners+=1 
            print('Outflow Wins: {} \nAzimuthal Wins: {}'.format(outflow_winners, azi_winners))
        if closest_to_planet:
            closest_azi = 30
            closest_outflow = 30
            for phi in phis:
                outflow_cross = data[phi][0]
                azi_cross = data[phi][1]
                if outflow_cross < closest_outflow:
                    closest_outflow = outflow_cross
                    closest_outflow_phi = phi
                if azi_cross < closest_azi:
                    closest_azi = azi_cross
                    closest_azi_phi = phi
            print('Closest Outflow Cross = {} at phi = {} \nClosest Azi Cross = {} at phi = {}'.format(closest_outflow, closest_outflow_phi, closest_azi, closest_azi_phi))
        if outflow_cross_plot:
            crosses = []
            phis_floats = [float(x) for x in phis]
            for phi in phis:
                outflow_cross = data[phi][0]
                crosses.append(outflow_cross)
            fig, ax = plt.subplots()
            ax.plot(phis_floats, crosses, label = 'Radial Distance Where $v_o > v_a$', color = 'k')
            ax.axhline(y =37.608040201005025, label = 'Radial Distance Where $v_o > v_a$ for Spin Aligned Dipole', color = 'r', linestyle = '--')
            ax.legend()
            ax.set_ylim(20,50)
            ax.set_xlabel('$\u03BB_{{III}} (Degrees)$')
            ax.set_ylabel('R ($R_J)$')
            plt.show()
        
        if furthest_azi_cross:
            furthest_azi = 10
            furthest_outflow = 10
            for phi in phis:
                azi_cross = data[phi][1]
                outflow_cross = data[phi][0]
                if outflow_cross > furthest_outflow:
                    furthest_outflow = outflow_cross
                    furthest_outflow_phi = phi
                if azi_cross > furthest_azi:
                    furthest_azi = azi_cross
                    furthest_azi_phi = phi
            print('Furthest azi Cross = {} at phi = {}'.format(furthest_azi, furthest_azi_phi))
            print('Furthest out Cross = {} at phi = {}'.format(furthest_outflow, furthest_outflow_phi))
        if footprint:
            footprints = []
            footprints_xy = []
            for phi_lh_deg in phis:
                print(phi_lh_deg)
                
                #outflow_cross = data[phi][0]
                azi_cross = data[phi_lh_deg][1]
                r = azi_cross
                phi_lh_deg = float(phi_lh_deg)
                phi_lh_rad = phi_lh_deg * np.pi/180
                phi_rh_rad = 2*np.pi - phi_lh_rad
                if self.aligned == 'yes':
                   colatitude = np.pi/2
                if self.aligned == 'no': 
                    cent_eq_latitude = self.centrifugal_equator(r, phi_rh_rad)
                    colatitude = np.pi/2 - cent_eq_latitude
                if footprint_direction == 'north':
                    footprint_point = self.trace_magnetic_field(starting_cordinates=[r*Rj, colatitude, phi_lh_rad], footprint=True, break_point=1.0, step = 0.001, pathing='backward', one_way='on')
                else:
                    footprint_point = self.trace_magnetic_field(starting_cordinates=[r*Rj, colatitude, phi_lh_rad], footprint=True, break_point=1.0, step = 0.001, one_way='on')
                footprints.append(footprint_point)
                footprints_xy.append([footprint_point[0],footprint_point[1]])
            
            footprints_plottable = np.transpose(footprints_xy)
            footprints_plottable_rj = footprints_plottable/Rj
            fig, ax = plt.subplots()
            ax.scatter(footprints_plottable_rj[0], footprints_plottable_rj[1], label = 'Footprint of Plasma Decoupling')
            #ax.add_patch(Circle((0,0), 1, color='firebrick', zorder=100, label = "Jupiter Radii", fill = False))
            ax.legend()
            np.save('{}_footprint_{}_new.npy'.format(model, footprint_direction), footprints_plottable_rj, allow_pickle=True)
            if show:
                plt.show()
            return footprints_plottable_rj
        if phi_cross_plot:
            phis = list(data.keys())
            phis_floats = [float(x) for x in phis]
            azi_crosses = []
            for phi in phis:
                azi_cross = data[phi][1]
                azi_crosses.append(azi_cross)
            if show:
                fig, ax = plt.subplots()
                ax.plot(phis_floats, azi_crosses)
                plt.show()
            return phis_floats, azi_crosses
        if outflow_footprint:
            footprints = []
            footprints_xy = []
            for phi_lh_deg in phis:
                print(phi_lh_deg)
                
                #outflow_cross = data[phi][0]
                outflow_cross = data[phi_lh_deg][0]
                r = outflow_cross
                phi_lh_deg = float(phi_lh_deg)
                phi_lh_rad = phi_lh_deg * np.pi/180
                phi_rh_rad = 2*np.pi - phi_lh_rad
                if self.aligned == 'yes':
                    colatitude = np.pi/2
                if self.aligned == 'no': 
                    cent_eq_latitude = self.centrifugal_equator(r, phi_rh_rad)
                    colatitude = np.pi/2 - cent_eq_latitude
                footprint_point = self.trace_magnetic_field(starting_cordinates=[r*Rj, colatitude, phi_lh_rad], footprint=True, break_point=1.0, step = 0.001)
                footprints.append(footprint_point)
                footprints_xy.append([footprint_point[0],footprint_point[1]])

            footprints_plottable = np.transpose(footprints_xy)
            footprints_plottable_rj = footprints_plottable/Rj
            fig, ax = plt.subplots()
            ax.plot(footprints_plottable_rj[0], footprints_plottable_rj[1], label = 'Footprint of Radial Outflow Decoupling')
            #ax.add_patch(Circle((0,0), 1, color='firebrick', zorder=100, label = "Jupiter Radii", fill = False))
            ax.legend()
            ax.set_aspect(aspect = 'equal')
            np.save('outflow_footprints.npy', footprints_plottable_rj, allow_pickle=True)
            if show:
                plt.show()
            return footprints_plottable_rj
    
    
    def db6_no_outflow(self, phi_lh_deg, numpoints = 200, stop = 60, corotation = False, ray = False, show = True, pensionerov = False):
        ''' for a given longitude, calculate radial outflow velocity vs local alfven velocity'''
        ''' first, all of the outflow velocities for all of the given mdots '''
        rs = np.linspace(6,stop,numpoints)
        #mdots = [500,1300,2000]

        ''' next, we need to calc alfven velocity, at each of the r values, along centrifigul equator, for given phi '''
        vas = []
        phi_rh_rad = (360 -phi_lh_deg) *np.pi/180
        phi_lh_rad = phi_lh_deg * np.pi/180
        corotations = []
        for r in rs:
            
            if self.aligned == 'no':
                cent_eq_latitude = self.centrifugal_equator(r, phi_rh_rad)
                colatitude = np.pi/2 - cent_eq_latitude
            else:
                colatitude = np.pi/2
            n = self.density_combined(r, colatitude, phi_lh_rad)
            B_overall = self.mag_field_at_point(r, colatitude, phi_rh_rad)
            B_x, B_y, B_z = self.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, colatitude, phi_rh_rad)
            B = np.array([B_x, B_y, B_z])
            B =  B/(10**9)  #chris's code is in nT
            va = self.calculator(B, n)
            vas.append(va)
            if corotation:
                corotations.append((2*np.pi/(9.93*3600))* r * Rj)

        

        ''' and then we plot ''' 
        fig, ax = plt.subplots()
        if ray:
            azimuthals = []
            ang_vel = np.load('angular_velocity_data/mdot_2000.0.npy', allow_pickle=True)
            ang_vel_rs = np.load('angular_velocity_data/radii2000.0.npy',  allow_pickle=True)
            for i in range(len(ang_vel_rs)):
                r = ang_vel_rs[i]
                omega = ang_vel[i]
                azimuthals.append(r*Rj*(omega_J-omega*omega_J))
            azimuthals_km = np.array(azimuthals)/1e3
            ax.plot(ang_vel_rs, azimuthals_km, label = 'Azimuthal Velocity (Hill Model)')

        vas_km = np.array(vas)/1000
        ax.plot(rs, vas_km, label = 'Local Alfven Velocity')
        if corotation:
            corotations_km = np.array(corotations)/1e3
            ax.plot(rs, corotations_km, label = 'Corotation Velocity', color = 'm')
        #ax.grid()
        if pensionerov:
            lt = self.what_LT_segment(phi_lh_deg=phi_lh_deg)
            with open('angular_velocity_data/Pensionerov_et_al/{}.txt'.format(lt), 'r') as f:
                data = f.read().splitlines()
            data = np.array([data[i].split() for i in range(len(data))])
            data = data.flatten()
            data = [data[i].replace(',', '') for i in range(len(data))]
            data = [float(data[i]) for i in range(len(data))]
            r_ang_vel = [data[i:i+2] for i in range(0, len(data), 2)]
            rs_vels = np.transpose(r_ang_vel)

            azis = []
            azis_rs = []
            for i in range(len(rs_vels[0])):
                r = rs_vels[0][i]
                omega = rs_vels[1][i]
                azis.append(r*Rj*(omega_J-omega*omega_J))
                #print(r, omega, omega_J, r*Rj*omega*omega_J )
                azis_rs.append(r)
            azis_km = np.array(azis)/1e3
            ax.plot(azis_rs, azis_km, label = 'Azimuthal Velocity (Pensionerov Model {})'.format(lt))
        ax.legend()
        ax.yaxis.set_ticks_position('both')
        plt.yscale("log")
        ax.set_ylim(1,1000)
        #ax.set_xlim(0,100)
        #ax.set(title = 'Radial Outflow Along Centrifugal Equator And Local Alfven Velocity \n For $\u03BB_{{III}} $ Longitude of {:.1f}{} '.format(phi_lh_deg, u"\N{DEGREE SIGN}"),
        ax.set(xlabel = 'R ($R_J$)',ylabel = 'Velocity ($kms^{-1}$)')
        ###plt.savefig('images-24-jan-update/radial_flow_plot_better.png')
        if show:
            plt.show()

    def is_mdot_enough(self, mdot, model = 'ray'):
        if model == 'ray':
            with open('crossing_data.json') as json_file:
                data = json.load(json_file)
        if model == 'pensionerov':
            with open('crossing_data_pensionerov.json') as json_file:
                data = json.load(json_file)
        #phis = np.linspace(0, 360, num = 360)
        phis = list(data.keys())
        success = False
        for phi in phis:
            print(phi)
            outflow_crossing = self.find_va_vo_cross_point([mdot], phi_lh_deg=float(phi), numpoints=200)
            outflow_crossing_r = outflow_crossing[mdot]
            azi_cross = data[phi][1]
            if outflow_crossing_r < azi_cross:
                success = True
                sphi = phi
                break

            
        if success:
            print('Success!!!! mdot = {} phi = {}'.format(mdot, sphi))

    def plot_va_vo_cross_points(self,numpoints = 360):
        phis = np.linspace(0, 360, num = numpoints)
        crossings = []
        for phi in phis:
            print('phi = ', phi)
            outflow_crossing = self.find_va_vo_cross_point([2000], phi_lh_deg=phi, numpoints=800)
            outflow_crossing_r = outflow_crossing[2000]
            crossings.append(outflow_crossing_r)
        fig, ax = plt.subplots()
        ax.plot(phis, crossings, label = 'Radial Distance Where $v_o >= v_a$', color = 'k') 
        ax.axhline(y =37.608040201005025, label = 'Radial Distance Where $v_o >= v_a$ for Spin Aligned Dipole', color = 'r', linestyle = '--')
        ax.legend()
        ax.set_ylim(20,50)
        ax.set_xlabel('$\u03BB_{{III}} (Degrees)$')
        ax.set_ylabel('R ($R_J$')
        plt.show()
start_time = time.time()
system = main('VIP4', 'no')
#system.density_topdown_contour(gridsize=60)
system.db6_no_outflow(200.8, 200, 60, True, True, True, True)
#system.analyse_cross_points(model = 'pensionerov', closest_to_planet=True)
#system.db6_no_outflow(200.8, corotation=True, ray=True, pensionerov=True)
#system.plot_angle_vs_time(num = 360, r = 20)
#system.difference_in_tt_multi(r = 20, num = 360)
'''
system.analyse_cross_points(model = 'ray', footprint=True, footprint_direction='north')
print('\n ONE DONE \n')
system.analyse_cross_points(model = 'ray', footprint=True, footprint_direction='south')
print('\n TWO DONE \n')
system.analyse_cross_points(model = 'pensionerov', footprint=True, footprint_direction='north')
print('\n THREE   DONE \n')
system.analyse_cross_points(model = 'pensionerov', footprint=True, footprint_direction='south')
'''

#print(system.find_furthest_r_single_input([10,0,np.pi/2]))
class comparisons:
    def __init__(self):
        self.dip = main('dipole', 'yes')
        self.vip = main('VIP4', 'no')
        self.aligned = main('VIP4', 'yes')

    def compare_B_radial_dip_vs_vip(self,phi):
        dip_results = self.dip.radial_profile_B_n(phi)
        vip_results = self.vip.radial_profile_B_n(phi)
        rs = dip_results[0]
        dipBs = dip_results[1]
        vipBs = vip_results[1]
        fig, ax = plt.subplots()
        ax.plot(rs, dipBs, label = 'Dipolar Field', color = 'r')
        ax.plot(rs, vipBs, label = 'VIP4', color = 'k')
        ax.set(xlabel = 'Radial Distance $(R_J)$', ylabel = 'Magnetic Field Strength (nT)')#, title = 'vip is super dip')
        ax.legend()
        plt.yscale("log")
        plt.show()


    def compare_va_along_field_lines(self, startpoint):
        start_copy = deepcopy(startpoint)
        start_copy_II = deepcopy(startpoint)
        dip_result = self.dip.va_along_field_line(startpoint)
    
        ali_result = self.aligned.va_along_field_line(start_copy_II)
        print(start_copy)
        vip_result = self.vip.va_along_field_line(start_copy)
        ss_dip, corrected_dip = dip_result[0], dip_result[1]
        ss_vip, corrected_vip = vip_result[0], vip_result[1]
        ss_ali, corrected_ali = ali_result[0], ali_result[1]
        corrected_dip_km = np.array(corrected_dip)/1e3
        corrected_vip_km = np.array(corrected_vip)/1e3
        corrected_ali_km = np.array(corrected_ali)/1e3
        fig, ax = plt.subplots()
        ax.plot(ss_dip, corrected_dip_km, label ='Spin Aligned Dipole', color = 'r')
        ax.plot(ss_vip, corrected_vip_km, label ='VIP4 Non Aligned Axes', color = 'k')
        ax.plot(ss_ali, corrected_ali_km, label = 'VIP4 Centrifugal and Spin Equators Aligned', color = 'teal')
        ax.set(xlabel = 'Distance Along Field Line $(R_J)$', ylabel = 'Alfven Velocity (kms$^{-1}$)', 
        title = 'Alfven Velocity Along field line \n Starting at ({:.0f},{:.1f},{:.1f})'.format(start_copy[0]/Rj, start_copy[1]*180/np.pi, 360 - start_copy[2]*180/np.pi))
        plt.legend()
        plt.show()

    def compare_va_distance_from_planet(self, startpoint, breakpoint = 2):
        direction = 'backward'
        start_copy = deepcopy(startpoint)
        start_copy_II = deepcopy(startpoint)
        dip_result = self.dip.va_along_field_line(startpoint, breakpoint= breakpoint, direction = direction)
    
        ali_result = self.aligned.va_along_field_line(start_copy_II, breakpoint= breakpoint, direction = direction)
        #print(start_copy)
        vip_result = self.vip.va_along_field_line(start_copy, breakpoint= breakpoint, direction = direction)
        rs_dip, corrected_dip = dip_result[3], dip_result[1]
        rs_vip, corrected_vip = vip_result[3], vip_result[1]
        rs_ali, corrected_ali =ali_result[3], ali_result[1]
        corrected_dip_km = np.array(corrected_dip)/1e3
        corrected_vip_km = np.array(corrected_vip)/1e3
        corrected_ali_km = np.array(corrected_ali)/1e3
        fig, ax = plt.subplots()
        ax.plot(rs_dip, corrected_dip_km, label ='Spin Aligned Dipole', color = 'r')
        ax.plot(rs_vip, corrected_vip_km, label ='VIP4 Non Aligned Axes', color = 'k')
        ax.plot(rs_ali, corrected_ali_km, label = 'VIP4 Centrifugal and Spin Equators Aligned', color = 'teal')
        ax.set(xlabel = 'Distance From Planet $(R_J)$', ylabel = 'Alfven Velocity (kms$^{-1}$)', 
        title = 'Alfven Velocity Along field line \n Starting at ({:.0f},{:.1f},{:.1f})\n against distance from planet'.format(start_copy[0]/Rj, start_copy[1]*180/np.pi, 360 - start_copy[2]*180/np.pi))
        plt.legend()
        plt.show()

    def compare_va_along_field_latitude(self, startpoint, breakpoint, limits = 'off', logplot = 'off', min_point = False):
        start_copy = deepcopy(startpoint)
        start_copy_II = deepcopy(startpoint)
        print(startpoint, start_copy_II, start_copy)
        dip_result = self.dip.va_along_field_line_both_directions(startpoint, breakpoint= breakpoint)
        #print(start_copy_II)
        ali_result = self.aligned.va_along_field_line_both_directions(start_copy_II, breakpoint= breakpoint)
        #print(start_copy)
        vip_result = self.vip.va_along_field_line_both_directions(start_copy, breakpoint= breakpoint)
        lats_dip, corrected_dip = dip_result[4], dip_result[1]
        lats_vip, corrected_vip = vip_result[4], vip_result[1]
        lats_ali, corrected_ali = ali_result[4], ali_result[1]
        corrected_dip_km = np.array(corrected_dip)/1e3
        corrected_vip_km = np.array(corrected_vip)/1e3
        corrected_ali_km = np.array(corrected_ali)/1e3
        fig, ax = plt.subplots()
        ax.plot(lats_dip, corrected_dip_km, label ='Spin Aligned Dipole', color = 'r')
        ax.plot(lats_vip, corrected_vip_km, label ='VIP4 Non Aligned Axes', color = 'k')
        ax.plot(lats_ali, corrected_ali_km, label = 'VIP4 Centrifugal and Spin Equators Aligned', color = 'teal')
        ax.set(xlabel = 'Latitude (Degrees)', ylabel = 'Alfven Velocity (kms$^{-1}$)') 
        #title = 'Alfven Velocity Along field line \n Passing Through at ({:.0f},{:.1f},{:.1f}) in eq plane'.format(start_copy[0]/Rj, start_copy[1]*180/np.pi, 360 - start_copy[2]*180/np.pi))
        if limits == 'on':
            ax.set_xlim(-10,10)
            ax.set_ylim(100,2000)
        if logplot == 'on':
            ax.set_yscale('log')

        if min_point:
            min_va_vip = np.amin(corrected_vip)
            min_index_vip = np.where(corrected_vip == min_va_vip)
            print(min_index_vip[0][0])
            min_lat_vip = lats_vip[min_index_vip[0][0]]
            print(' vip min lat = ', min_lat_vip)

        plt.legend()
        plt.show()

    def compare_n_along_field_line(self, startpoint, limits = 'off', logplot = 'off', breakpoint = 6, max_point = True):
        start_copy = deepcopy(startpoint)
        start_copy_II = deepcopy(startpoint)
        dip_result = self.dip.va_along_field_line_both_directions(startpoint, breakpoint= breakpoint)
        #print(start_copy_II)
        ali_result = self.aligned.va_along_field_line_both_directions(start_copy_II, breakpoint= breakpoint)
        #print(start_copy)
        vip_result = self.vip.va_along_field_line_both_directions(start_copy, breakpoint= breakpoint)
        lats_dip, n_dip = dip_result[4], dip_result[7]
        lats_vip, n_vip = vip_result[4], vip_result[7]
        lats_ali, n_ali = ali_result[4], ali_result[7]
        n_dip_cm = np.array(n_dip)/1e6
        n_vip_cm = np.array(n_vip)/1e6
        n_ali_cm = np.array(n_ali)/1e6
        fig, ax = plt.subplots()
        ax.plot(lats_dip, n_dip_cm, label ='Spin Aligned Dipole', color = 'r')
        ax.plot(lats_vip, n_vip_cm, label ='VIP4 Non Aligned Axes', color = 'k')
        ax.plot(lats_ali, n_ali_cm, label = 'VIP4 Centrifugal and Spin Equators Aligned', color = 'teal')
        ax.set(xlabel = 'Latitude (Degrees)', ylabel = 'Density (cms$^{-1}$)') 
        #title = 'Alfven Velocity Along field line \n Passing Through at ({:.0f},{:.1f},{:.1f}) in eq plane'.format(start_copy[0]/Rj, start_copy[1]*180/np.pi, 360 - start_copy[2]*180/np.pi))
        if limits == 'on':
            ax.set_xlim(-20,20)
            #ax.set_ylim(0,2000)
        if logplot == 'on':
            ax.set_yscale('log')

        plt.legend()
        plt.show()
        
        if max_point:
            max_n_vip = np.amax(n_vip_cm)
            max_index_vip = np.where(n_vip_cm == max_n_vip)
            print(max_index_vip[0][0])
            min_lat_vip = lats_vip[max_index_vip[0][0]]
            print(' vip max lat = ', min_lat_vip)
            print('vip max n =', max_n_vip)
            print('\n\n')

            max_n_dip = np.amax(n_dip_cm)
            max_index_dip = np.where(n_dip_cm == max_n_dip)
            print(max_index_dip[0][0])
            min_lat_dip = lats_dip[max_index_dip[0][0]]
            print('dip max lat = ', min_lat_dip)
            print('dip max n =', max_n_dip)


    def compare_dist_from_planet_lat(self,startpoint,  limits = 'off', logplot = 'off', breakpoint = 6):
        start_copy = deepcopy(startpoint)
        start_copy_II = deepcopy(startpoint)
        start_copy_II[2] = 2*np.pi - start_copy_II[2]
        dip_result = self.dip.va_along_field_line_both_directions(startpoint, breakpoint= breakpoint)
        #print(start_copy_II)
        
        #print(start_copy)
        vip_result = self.vip.va_along_field_line_both_directions(start_copy, breakpoint= breakpoint)
        lats_dip, r_dip = dip_result[4], dip_result[3]
        lats_vip, r_vip = vip_result[4], vip_result[3]

        #r_dip_rj = np.array(r_dip)/Rj
        #r_vip_rj = np.array(r_vip)/Rj
        fig, ax = plt.subplots()
        ax.plot(lats_dip, r_dip, label ='Spin Aligned Dipole', color = 'r')
        ax.plot(lats_vip, r_vip, label ='VIP4', color = 'k')
        #asplt.axvline(x = -7.360633237444866, color = 'teal', linestyle = '--', label = 'VIP4 Field Line Intersects Centrifugal Plane')
        #ax.plot(lats_ali, r_ali, label = 'VIP4 Centrifugal and Spin Equators Aligned', color = 'teal')
        ax.set(xlabel = 'Latitude (Degrees)', ylabel = 'Distance From Planet ($R_J$)') 
        #title = 'Alfven Velocity Along field line \n Passing Through at ({:.0f},{:.1f},{:.1f}) in eq plane'.format(start_copy[0]/Rj, start_copy[1]*180/np.pi, 360 - start_copy[2]*180/np.pi))
        if limits == 'on':
            ax.set_xlim(-20,20)
            ax.set_ylim(0,4e3)
        if logplot == 'on':
            ax.set_yscale('log')

        plt.legend()
        plt.show()

    def compare_alfven_at_point(self, point):
        '''
        r in rj 
        theta in colatitude deg 
        phi in lh deg
        '''
        aligned = self.aligned.alfven_at_point(point)
        dipole = self.dip.alfven_at_point(point)
        vip4 = self.vip.alfven_at_point(point)
        print('aligned = {} \n dipole = {},  \n vip4 = {}'.format(aligned, dipole, vip4))
        print(dipole/vip4)

    def compare_models_footprints_calc(self, direction = 'south'):
        if direction == 'south':
            ray = self.vip.analyse_cross_points(model = 'ray', footprint=True, footprint_direction='north')
            pensionerov = self.vip.analyse_cross_points(model= 'pensionerov', footprint=True, footprint_direction='north')
            fig, ax = plt.subplots()
            ax.plot(ray[0], ray[1], label = 'Ray Model', color = 'k')
            ax.plot(pensionerov[0], pensionerov[1], label = 'Pensionerov Model', color = 'r')
            ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter", fill = False))
            ax.set(xlabel = 'x $(R_J)$', ylabel = 'y $(R_J)$')
            np.save('pensionerov_footprint_south.npy', pensionerov, allow_pickle=True)
            np.save('ray_footprint_north_south.npy', ray, allow_pickle= True)
        else:
            ray = self.vip.analyse_cross_points(model = 'ray', footprint=True)
            pensionerov = self.vip.analyse_cross_points(model= 'pensionerov', footprint=True)
            fig, ax = plt.subplots()
            np.save('pensionerov_footprint.npy', pensionerov, allow_pickle=True)
            np.save('ray_footprint.npy', ray, allow_pickle= True)
            ax.plot(ray[0], ray[1], label = 'Ray Model', color = 'k')
            ax.plot(pensionerov[0], pensionerov[1], label = 'Pensionerov Model', color = 'r')
            ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter", fill = False))
            ax.set(xlabel = 'x $(R_J)$', ylabel = 'y $(R_J)$')
            np.save('pensionerov_footprint.npy', pensionerov, allow_pickle=True)
            np.save('ray_footprint.npy', ray, allow_pickle= True)
        plt.show()


    
    def compare_models_crosses(self):
        ray = self.vip.analyse_cross_points(model = 'ray', phi_cross_plot=True)
        pensionerov = self.vip.analyse_cross_points(model= 'pensionerov', phi_cross_plot=True)
        fig, ax = plt.subplots()
        ax.plot(ray[0], ray[1], label = 'Hill Model', color = 'k')
        ax.plot(pensionerov[0], pensionerov[1], label = 'Pensionerov Model', color = 'r')
        ax.legend()
        ax.set(ylabel = 'Distance Where $v_{\u03A6} > v_A$ ($R_J)$', xlabel = '$\u03BB_{{III}}$ (Degrees)')
        plt.show()

    def compare_models_footprints_pre_calculated(self):
        ray = np.load('ray_footprint.npy', allow_pickle=True)
        pensionerov = np.load('pensionerov_footprint.npy', allow_pickle=True)
        fig, ax = plt.subplots()
        ax.plot(ray[0], -ray[1], label = 'Ray Model', color = 'k')
        ax.plot(pensionerov[0], pensionerov[1], label = 'Pensionerov Model', color = 'r')
        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter", fill = False))
        ax.set(xlabel = 'x $(R_J)$', ylabel = 'y $(R_J)$')
        ax.legend()
        ax.set_aspect(aspect='equal')
        plt.show()
    
    def compare_all_decoupling_pre_calculated(self):
        #ray = np.load('ray_footprint_south_new.npy', allow_pickle=True)
        pensionerov = np.load('pensionerov_footprint_south_new.npy', allow_pickle=True)
        #outflow = np.load('outflow_footprints.npy', allow_pickle=True)
        fig, ax = plt.subplots()
        #ax.plot(ray[0], -ray[1], label = 'Ray Model', color = 'k')
        ax.plot(pensionerov[0], -pensionerov[1], label = 'Footprint of Plasma Decoupling', color = 'r')
        #ax.plot(outflow[0], outflow[1], label = 'Radial Outflow Decoupling', color = 'c')
        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter", fill = False))
        ax.set(xlabel = 'x $(R_J)$', ylabel = 'y $(R_J)$')
        ax.legend()
        ax.set_aspect(aspect='equal')
        plt.show()
comparisons = comparisons()
comparisons.compare_models_crosses()
#comparisons.compare_all_decoupling_pre_calculated()
#comparisons.compare_models_crosses()
#comparisons.compare_all_decoupling_pre_calculated()
#comparisons.compare_all_decoupling_pre_calculated()
#comparisons.compare_models_footprints_pre_calculated()
#comparisons.compare_n_along_field_line(logplot='on', startpoint=[10, np.pi/2, 290.8 *np.pi/180])
#comparisons.compare_models_crosses()
#omparisons.compare_models_footprints_calc(direction='north')
#comparisons.compare_B_radial_dip_vs_vip(110.8)
#comparisons.compare_B_radial_dip_vs_vip(200.8)
#comparisons.compare_va_distance_from_planet([10, np.pi/2, 200.8*np.pi/180], breakpoint = 2)
comparisons.compare_va_along_field_latitude([10, np.pi/2, 200.8*np.pi/180], breakpoint = 6, limits='off', min_point=True, logplot='on')
#comparisons.compare_alfven_at_point([10, 90, 290.8])
#comparisons.compare_n_along_field_line([10, np.pi/2, 200.8*np.pi/180], logplot='on')
#comparisons.compare_dist_from_planet_lat([10, np.pi/2, 200.8*np.pi/1)

print('time taken = {}s'.format(time.time()-start_time))