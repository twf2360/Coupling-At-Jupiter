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
from density_height import DensityHeight
from radial_outflow import radialOutflow
from matplotlib import ticker, cm
from field_and_current_sheet import InternalAndCS
Rj = 7.14 * (10 ** 7)
mu_0 = 1.25663706212 * 10 ** -6

class AlfvenVel:
    def __init__(self, avgIonMass =28,numpoints= 5000, start = 0, stop = 30, model = 'VIP4',):
        self.start =start
        self.model = model
        self.stop = stop 
        self.numpoints = numpoints
        self.avgIonMass = avgIonMass * 1.67 * 10**-27
        self.help = HelpfulFunctions()
        self.densityfunctions = DensityHeight(self.numpoints, self.start, self.stop)
        self.gridx, self.gridy = self.help.makegrid_2d_negatives(self.numpoints ,gridsize= self.stop)
        self.radialfunctions = radialOutflow(self.avgIonMass)
        self.field = field_models()
        self.plotter = InternalAndCS()

    def calculator(self, B, n):
        magB = np.linalg.norm(B)
        rho = n * self.avgIonMass 
        Va = magB/np.sqrt((rho * mu_0))
        #print('b = {}, va = {}, rho = {}, magB = {}'.format(B, Va, rho, magB))
        return Va

    #add a constant phi part too! 
    
    def top_down_matched_equators(self):
        theta = np.pi/2 #<- CHANGE THIS TO VIEW A SLIGHTLY DIFFERENT PLANE
        x_s = []
        y_s = []
        spacing = self.stop/self.numpoints
        for i in range(self.numpoints):
            x_s.append(i * spacing)
            y_s.append(i* spacing)
            x_s.append(-i * spacing)
            y_s.append(-i* spacing)
        x_s.sort()
        y_s.sort()
        n_0s = []
        #print(x_s, y_s)

        Vas = []
        for i in range(len(y_s)):
            print('new row, {} to go'.format(len(y_s)-i))
            Vas_row = [] 
            for j in range(len(x_s)):
                r = np.sqrt((x_s[j])**2 + (y_s[i])**2)
                
                #print(r)
                if r <6:
                    va = 1 *10 ** 2
                    Vas_row.append(va)
                    continue
                n = self.radialfunctions.radial_density(abs(r))
                phi = np.arctan2(y_s[i],x_s[j])
                B_r, B_theta, B_phi = self.field.Internal_Field(r, theta, phi, model=self.model) #calculates the magnetic field due to the internal field in spherical polar that point)
                B_current = self.field.CAN_sheet(r, theta, phi) #calculates the magnetic field due to the current sheet in spherical polar
                B_notcurrent = np.array([B_r, B_theta, B_phi]) 
                B_overall = np.add(B_current, B_notcurrent) #adds up the total magnetic field 
                B_x, B_y, B_z = self.help.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi)
                B = np.array([B_x, B_y, B_z])
                B =  B/(10**9)  #chris's code is in nT
                va = self.calculator(B, n)
                Vas_row.append(va)


            Vas.append(Vas_row)

        Vas_km = np.array(Vas)/(1000)

        #log_vas_km = np.log(Vas_km)
        fig, ax = plt.subplots()
        cont = ax.contourf(self.gridx, self.gridy, Vas_km, cmap = 'bone')#, locator=ticker.LogLocator())
        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        ax.add_patch(Circle((0,0), 6, color='c', zorder=90, label = "Io Orbital Radius"))
        ax.legend()
        ax.set_xlim(-30,30)
        ax.set_ylim(-30,30)
        degrees = theta * 180 /np.pi
        ax.set(xlabel = 'x $R_J$ \n CML is vertically upwards', ylabel = 'y $R_J$', title = 'alfven velocity in the colatitude = {:.0f}{} plane'.format(degrees, u"\N{DEGREE SIGN}"))
        fig.colorbar(cont, label = '$V_a km$')
        ax.set_aspect('equal', adjustable = 'box')
        plt.savefig('images/va_topdown.png')
        plt.show() 

    def topdown_seperate_equators(self, spin_eq = 'on', density = 'off'):
        '''
        top down view from the spin equator - could be extended to do magnetic field eq too later on. 
        just takes into account the difference in density that will arise due to the different centrifugal equator
        '''
        if spin_eq == 'on':
            if density == 'on':
                self.spin_eq_topdown(density='on')
            else:
                self.spin_eq_topdown()
    
    def spin_eq_topdown(self, density = 'off'): #knowing what we now know about the co-ordinate transform you may want to check this! 
        '''  
        does the actual stuff for the topdown centrifugal equators thing
        '''
        theta = np.pi/2 #<- CHANGE THIS TO VIEW A SLIGHTLY DIFFERENT PLANE
        x_s = []
        y_s = []
        spacing = self.stop/self.numpoints
        for i in range(self.numpoints):
            x_s.append(i * spacing)
            y_s.append(i* spacing)
            x_s.append(-i * spacing)
            y_s.append(-i* spacing)
        x_s.sort()
        y_s.sort()
        n_0s = []
        #print(x_s, y_s)

        Vas = []
        density_list = []
        for i in range(len(y_s)):

            print('new row, {} to go'.format(len(y_s)-i))
            Vas_row = [] 
            density_row = []
            for j in range(len(x_s)):
                r = np.sqrt((x_s[j])**2 + (y_s[i])**2)
                phi = np.arctan2(y_s[i],x_s[j])
                

                
                #print(r)
                if r <6:
                    va = 1 *10 ** 2
                    Vas_row.append(va)
                    continue
                r_cent = r 
                phi_cent = phi
                theta_shift = self.help.centrifugal_equator(r, phi)
                theta_cent = theta + theta_shift
                
                scaleheight = self.densityfunctions.scaleheight(r_cent)
                n_0 = self.radialfunctions.radial_density(r_cent)
                x_cent, y_cent, z_cent = self.help.sph_to_cart(r_cent, theta_cent, phi_cent)
                den = self.densityfunctions.density(n_0, z_cent, scaleheight)
                
 
                
                B_r, B_theta, B_phi = self.field.Internal_Field(r, theta, phi, model=self.model) #calculates the magnetic field due to the internal field in spherical polar that point)
                B_current = self.field.CAN_sheet(r, theta, phi) #calculates the magnetic field due to the current sheet in spherical polar
                B_notcurrent = np.array([B_r, B_theta, B_phi]) 
                B_overall = np.add(B_current, B_notcurrent) #adds up the total magnetic field 
                B_x, B_y, B_z = self.help.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi)
                B = np.array([B_x, B_y, B_z])
                B = B/(10**9) #CHRIS code outputs nT
                va = self.calculator(B, den)
                Vas_row.append(va)
                density_row.append(den)
                
            #print(density_row)
            Vas.append(Vas_row)
            density_list.append(density_row)


        Vas_km = np.array(Vas)/(1000)
        densities = np.array(density_list)
        print(densities[0], '\n', densities[1])  
        
        print(densities.shape, '\n', Vas_km.shape)
        fig, ax = plt.subplots()
        cont = ax.contourf(self.gridx, self.gridy, Vas_km, cmap = 'bone')#, locator=ticker.LogLocator())
        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        ax.add_patch(Circle((0,0), 6, color='c', zorder=90, label = "Io Orbital Radius"))
        ax.legend()
        ax.set_xlim(-30,30)
        ax.set_ylim(-30,30)
        ax.set(xlabel = 'x $R_J$ \n CML is vertically upwards', ylabel = 'y $R_J$', title = 'alfven velocity in the spin plane, taking centrifugal equator into account')
        fig.colorbar(cont, label = '$V_a km$')
        ax.set_aspect('equal', adjustable = 'box')
        plt.savefig('images/va_topdown_inc_cent.png')
        plt.show() 
        if density == 'on':
        
            fig, ax = plt.subplots()
            cont = ax.contourf(self.gridx, self.gridy, densities, cmap = 'bone')#, locator=ticker.LogLocator())
            ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
            ax.add_patch(Circle((0,0), 6, color='c', zorder=90, label = "Io Orbital Radius"))
            ax.legend()
            ax.set_xlim(-30,30)
            ax.set_ylim(-30,30)
            ax.set(xlabel = 'x $R_J$ \n CML is vertically upwards', ylabel = 'y $R_J$', title = 'density in the spin plane, taking centrifugal equator into account')
            fig.colorbar(cont, label = 'Density $cm^-3$')
            ax.set_aspect('equal', adjustable = 'box')
            plt.savefig('images/va_topdown_inc_cent.png')
            plt.show() 


    def sideview_seperate_equators(self):
        ''' 
        Plots the alfven velocity for a side on view 
        ''' 
        phi = 0 
        y_s = []
        z_s = []
        spacing = self.stop/self.numpoints
        for i in range(self.numpoints):
            y_s.append(i * spacing)
            z_s.append(i* spacing)
            y_s.append(-i * spacing)
            z_s.append(-i* spacing)
        y_s.sort()
        z_s.sort()
    
        n_0s = []
        Vas = []

        for i in range(len(z_s)):
            print('new row, {} to go'.format(len(z_s)-i))
            Vas_row = [] 
            for j in range(len(y_s)):
                r = np.sqrt((y_s[j])**2 + (z_s[i])**2)
                theta = np.arctan2(y, z)
                HeightAboveCent = self.help.height_centrifugal_equator(r, phi)
                r_cent = self.help.length_centrifual_equator(r, phi)
                scaleheight = self.densityfunctions.scaleheight(r_cent)
                

                
                #print(r)
                if r <6:
                    va = 1 *10 ** 2
                    Vas_row.append(va)
                    continue
                n_0 = self.radialfunctions.radial_density(abs(r_cent))
                n = self.densityfunctions.density(n_0, HeightAboveCent, scaleheight) 
                ''' 
                this is wrong as it doesn't include the height above the spin equator, just the height of the spin equator above the centrifigual equator!
                ''' 
                B_r, B_theta, B_phi = self.field.Internal_Field(r, theta, phi, model=self.model) #calculates the magnetic field due to the internal field in spherical polar that point)
                B_current = self.field.CAN_sheet(r, theta, phi) #calculates the magnetic field due to the current sheet in spherical polar
                B_notcurrent = np.array([B_r, B_theta, B_phi]) 
                B_overall = np.add(B_current, B_notcurrent) #adds up the total magnetic field 
                B_x, B_y, B_z = self.help.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi)
                B = np.array([B_x, B_y, B_z])
                B = B/(10**9) #CHRIS code outputs nT
                va = self.calculator(B, n)
                Vas_row.append(va)

            Vas.append(Vas_row)
        Vas_km = np.array(Vas)/(1000)

        fig, ax = plt.subplots()
        cont = ax.contourf(self.gridx, self.gridy, Vas_km, cmap = 'bone')#, locator=ticker.LogLocator())
        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        ax.add_patch(Circle((0,0), 6, color='c', zorder=90, label = "Io Orbital Radius"))
        ax.legend()
        ax.set_xlim(-30,30)
        ax.set_ylim(-30,30)
        ax.set(xlabel = 'y $R_J$ \n CML is out of screen', ylabel = 'z $R_J$', title = 'alfven velocity in the plane perpendicular to CML, taking centrifugal equator into account')
        fig.colorbar(cont, label = '$V_a km$')
        ax.set_aspect('equal', adjustable = 'box')
        plt.savefig('images/va_side_inc_cent.png')
        plt.show() 


    def calc_3d(self, r):
        spacing = 2 * np.pi / self.numpoints
        points_sph = []
        points_cart = []
        thetas = []
        phis = []
        n_0 = self.radialfunctions.radial_density(r*Rj)
        Vas = []
        for i in range(self.numpoints):
            phi = i * spacing
            theta = i * spacing
            thetas.append(theta)
            phis.append(phi)
            #points_sph.append([r, theta, phi])
        for theta in thetas:
            for phi in phis:
                points_sph.append([r, theta, phi])

        for point in points_sph:
            px, py, pz = self.help.sph_to_cart(point[0], point[1], point[2])
            points_cart.append([px,py,pz])

            H = self.densityfunctions.scaleheight(r*Rj)
            z = r*np.cos(theta)
            n = self.densityfunctions.density(n_0, z, H)

            B_r, B_theta, B_phi = self.field.Internal_Field(r, point[1], point[2], model=self.model) #calculates the magnetic field due to the internal field in spherical polar that point)
            B_current = self.field.CAN_sheet(r, point[1], point[2]) #calculates the magnetic field due to the current sheet in spherical polar
            B_notcurrent = np.array([B_r, B_theta, B_phi]) 
            B_overall = np.add(B_current, B_notcurrent) #adds up the total magnetic field 
            B_cart = self.help.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r*Rj, point[1], point[2])
            va = self.calculator(B_cart, n)
            Vas.append(va)
        points_cart = np.array(points_cart)
        log_vas = np.log(Vas)

        #make a straight line through the figure so it makes a bit more sense
        x = [0,0]
        y = [0,0]
        z = [-40, 40]
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plottable = np.transpose(points_cart)
        scatter = ax.scatter(plottable[0], plottable[1], plottable[2], c=log_vas, cmap = 'bone')
        ax.plot(x,y,z, color = 'black', label = 'Spin axis')

        fig.colorbar(scatter, label = 'ln($V_a$)')
        ax.set_xlim3d(-40, 40)
        ax.set_ylim3d(-40, 40)
        ax.set_zlim3d(-40, 40)
        ax.set_xlabel('$X, R_j$', fontsize=10)
        ax.set_ylabel('$Y, R_J$', fontsize=10)
        ax.legend()
        plt.show()
    
        
    def travel_time(self, startpoint = [30, np.pi/2, 0]):
        '''
        Calculate the travel path/time of an alfven wave from a given startpoint to the ionosphere. 
        input startpoint [r, theta, phi] where r is in rj and phi is left handed
        ''' 
        startpoint[0] = startpoint[0]*Rj
        plot_results = self.plotter.trace_magnetic_field(starting_cordinates=startpoint, one_way='on')
        points = np.array(plot_results[0])
        Bs = np.array(plot_results[2]) /10e9
        ''' 
        this returns the path taken (in terms of point by point) taken by the alfven wave (points)
        and the magnetic field at each points (Bs)
        '''  
        time = 0 
        for i in range(len(points)-1):
            start_point = points[i]
            end_point = points[i+1]
            difference = end_point - start_point
            distance = np.linalg.norm(difference)
            midpoint = end_point - difference/2
            
            B_start = Bs[i]
            B_end = Bs[i+1]
            magB_start = np.linalg.norm(B_start)
            magB_end = np.linalg.norm(B_end)
            averageB = (magB_end + magB_start)/2

            ''' 
            - this part is probably worth checking with Licia/someone
            - approximates the alfven velocity by calculating B as halfway between B_start  and B_end 
                and by calculating n half way between the two points
            - will also have to add in the approximate function for n inside of Io radius
            '''
            r, theta, phi = self.help.cart_to_sph(midpoint[0], midpoint[1], midpoint[2])
            r_cent = r 
            phi_cent = phi
            theta_shift = self.help.centrifugal_equator(r, phi)
            theta_cent = theta + theta_shift
            
            scaleheight = self.densityfunctions.scaleheight(r_cent)
            n_0 = self.radialfunctions.radial_density(r_cent)
            x_cent, y_cent, z_cent = self.help.sph_to_cart(r_cent, theta_cent, phi_cent)
            n = self.densityfunctions.density(n_0, z_cent, scaleheight)

            va = self.calculator(averageB, n)
            '''  
            -later plan to improve this to a proper relativistic correction
            ''' 
            if va > 3e8:
                va=0.9 * 3e8
            traveltime = distance/va
            time += traveltime
        print('travel time = {:.2f}s (= {:.1f}mins)'.format(time, time/60))
        return time


            
        


            


test = AlfvenVel(numpoints=200)
#test.top_down_matched_equators()
test.topdown_seperate_equators(density = 'on')
test.travel_time([30, np.pi/2, 212* np.pi/180])

                


        
