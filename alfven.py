import math
import numpy as np
import sys
import matplotlib.pyplot as plt
import json
from helpful_functions import HelpfulFunctions
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, PathPatch
from matplotlib.lines import Line2D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib as mpl
from mag_field_models import field_models
import scipy
import matplotlib.colors as mcolors
import scipy.special
from density_height import DensityHeight
from radial_outflow import radialOutflow
from matplotlib import ticker, cm
from field_and_current_sheet import InternalAndCS
plt.rcParams.update({'font.size': 22})
plt.rcParams['legend.fontsize'] = 14
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
                va_corrected = self.relativistic_correction(va)
                Vas_row.append(va_corrected)


            Vas.append(Vas_row)

        Vas_km = np.array(Vas)/(1000)

        #log_vas_km = np.log(Vas_km)
        fig, ax = plt.subplots(figsize = (25,15))
        cont = ax.contourf(self.gridx, self.gridy, Vas_km, cmap = 'bone')#, locator=ticker.LogLocator())
        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        ax.add_patch(Circle((0,0), 6, color='c', zorder=90, label = "Io Orbital Radius"))
        ax.legend()
        ax.set_xlim(-30,30)
        ax.set_ylim(-30,30)
        degrees = theta * 180 /np.pi
        ax.set(xlabel = 'x $(R_J)$', ylabel = 'y $(R_J)$', title = 'alfven velocity in the colatitude = {:.0f}{} plane'.format(degrees, u"\N{DEGREE SIGN}"))
        fig.colorbar(cont, label = '$V_a (km)$')
        ax.set_aspect('equal', adjustable = 'box')
        plt.savefig('images-24-jan-update/va_topdown.png')
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
                    n = 1e6
                    density_row.append(n)
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
                va_corrected = self.relativistic_correction(va)
                Vas_row.append(va_corrected)
                density_row.append(den)
                
            #print(density_row)
            Vas.append(Vas_row)
            density_list.append(density_row)


        Vas_km = np.array(Vas)/(1000)
        densities = np.array(density_list)
        #print(densities[0], '\n', densities[1])  
        
        #print(densities.shape, '\n', Vas_km.shape)
        fig, ax = plt.subplots()
        cont = ax.contourf(self.gridx, self.gridy, Vas_km, cmap = 'bone')#, locator=ticker.LogLocator())
        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        ax.add_patch(Circle((0,0), 6, color='c', zorder=90, label = "Io Orbital Radius"))
        ax.legend()
        ax.set_xlim(-30,30)
        ax.set_ylim(-30,30)
        ax.set(xlabel = 'x $(R_J)$ ', ylabel = 'y $(R_J)$', title = 'alfven velocity in the spin plane, taking centrifugal equator into account')
        fig.colorbar(cont, label = '$V_a (km)$')
        ax.set_aspect('equal', adjustable = 'box')
        plt.savefig('images-24-jan-update/va_topdown_inc_cent.png')
        plt.show() 
        if density == 'on':
        
            fig, ax = plt.subplots()
            cont = ax.contourf(self.gridx, self.gridy, densities, cmap = 'bone', locator=ticker.LogLocator())
            ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
            ax.add_patch(Circle((0,0), 6, color='c', zorder=90, label = "Io Orbital Radius"))
            ax.legend()
            ax.set_xlim(-30,30)
            ax.set_ylim(-30,30)
            ax.set(xlabel = 'x $(R_J)$', ylabel = 'y $(R_J)$', title = 'density in the spin plane, taking centrifugal equator into account')
            fig.colorbar(cont, label = 'Density $(cm^-3)$')
            ax.set_aspect('equal', adjustable = 'box')
            plt.savefig('images-24-jan-update/va_topdown_inc_cent.png')
            plt.show() 


    def sideview_seperate_equators(self,phi_lh, lines = 'off'):

        ''' 
        plots a slice of the alfven velocity at a certain longitude given by phi_lh (in degrees)
        '''
        phi_rh = phi_lh#2*np.pi - (phi_lh * np.pi/180) 
        ''' lets mess with this a bit'''
        #print(phi_rh)
        densities = []
        grids, gridz = self.help.makegrid_2d_negatives(200 ,gridsize= self.stop)

        r_cent_points = np.linspace(-30, 30, num=200)
        cent_plot_points = []
        for point in r_cent_points:
            if point < 0:
                phi = phi_rh + np.pi 
            else: 
                phi = phi_rh


            r_cent, theta_cent, phi_cent = self.help.change_equators(point, np.pi/2, phi)
            z_cent = abs(point) * np.cos(theta_cent)

            cent_plot_points.append([point, -z_cent]) 
        cent_plot_points = np.array(cent_plot_points)

        cent_plot_points_t = np.transpose(cent_plot_points)
        

        Vas = []
        for i in range(len(gridz)):
            print('new row, {} to go'.format(len(gridz)-i))
            Vas_row = []

            for j in range(len(grids)):
                z = gridz[i][j]
                s = grids[i][j]

                r = np.sqrt(z**2 + s**2)
                phi = phi_rh
                theta = np.arctan2(s,z)
               
                if r < 6:
                    va = 1e6
                    Vas_row.append(va)
                    continue
                    
                n = self.densityfunctions.density_sep_equators(r, theta, phi)
                if n < 1e4: 
                    n = 1e4
                    ''' CURRENT LOW DENSITY CORRECTION'''
                #print(s, z, theta, phi)
                
                #print(n)
                ''' NOTE:THERE'S NOT A PHI SWITCH HERE FOR WHICH SIDE YOU'RE ON! '''
                B_r, B_theta, B_phi = self.field.Internal_Field(r, theta, phi, model=self.model) #calculates the magnetic field due to the internal field in spherical polar that point)
                B_current = self.field.CAN_sheet(r, theta, phi) #calculates the magnetic field due to the current sheet in spherical polar
                B_notcurrent = np.array([B_r, B_theta, B_phi]) 
                B_overall = np.add(B_current, B_notcurrent) #adds up the total magnetic field 
                B_x, B_y, B_z = self.help.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi)
                B = np.array([B_x, B_y, B_z])
                B = B/(1e9) #CHRIS code outputs nT
                va = self.calculator(B, n)
                va_corrected = self.relativistic_correction(va)
                Vas_row.append(va_corrected)
            Vas.append(Vas_row)
            
        vas_km = np.array(Vas)/1e3
        #print(Vas)

        fig, ax = plt.subplots(figsize = (25,16))
        lev_exp = np.arange(np.floor(np.log10(vas_km.min())-1), np.ceil(np.log10(vas_km.max())+1), step = 0.25)
        levs = np.power(10, lev_exp)
        cont = ax.contourf(grids, gridz, vas_km, cmap = 'bone', levels = levs, norm=mcolors.LogNorm())# locator=ticker.LogLocator())
        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        ax.add_patch(Circle((0,0), 6, color='c', zorder=90, label = "Io Orbital Radius"))
        ax.text(0.95, 0.01, 'SYS III (lh) Longitutude = {} '.format(phi_lh),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='k', fontsize=15)
        ax.set_xlim(-30,30)
        ax.set_ylim(-15,15)
        ax.set(xlabel = '$R_J$ \n', ylabel = '$R_J$', title = 'Meridian Slice')
        if lines == 'on':
            ax.plot(cent_plot_points_t[0], cent_plot_points_t[1], label = 'Centrifugal Equator')
        fig.colorbar(cont, label = 'V$_a$ $(kms^{-1})$')
        ax.set_aspect('equal', adjustable = 'box')
        ax.legend()
        plt.savefig('images-24-jan-update/va side slice.png')
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
            va = self.calculator(B_cart/1e9, n)
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
    
        
    def travel_time(self, startpoint = [30, np.pi/2, 212* np.pi/180], direction = 'forward', path_plot = 'off', dr_plot = 'off', print_time = 'on', va_plot ='off',b_plot = 'off', n_plot = 'off', 
    debug_plot = 'off'):
        '''
        Calculate the travel path/time of an alfven wave from a given startpoint to the ionosphere. 
        input startpoint [r, theta, phi] where r is in rj and phi is left handed
        ''' 
        startpoint[0] = startpoint[0]*Rj
        #startpoint[2] = 2*np.pi - startpoint[0]
        phi_lh = startpoint[2]

        plot_results = self.plotter.trace_magnetic_field(starting_cordinates=startpoint, one_way='on', break_point=2, step = 0.001, pathing= direction)

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
            

            ''' 
            - this part is probably worth checking with Licia/someone (espicially as it keeps returning a time that is too small)
            - approximates the alfven velocity by calculating B as halfway between B_start  and B_end 
                and by calculating n half way between the two points
            - will also have to add in the approximate function for n inside of Io radius
            '''
            r, theta, phi = self.help.cart_to_sph(midpoint[0], midpoint[1], midpoint[2])
            r = r/Rj
            
            if r < 6: 
                '''
                this is where the density problem lies - we need to put a better version of the density in here! 
                ''' 
                '''
                va = 0.9 * 3e8
                traveltime = distance/va
                time += traveltime
                continue
                '''
                n = 4711117 *np.exp(r - 6)
                
                va = self.calculator(averageB, n)
                #print(averageB, n)
                va_uncorrected_list.append(va)

                va_corrected = self.relativistic_correction(va)
                va_corrected_list.append(va)
                print(n, r, va, va_corrected)
                
                
                traveltime = distance/va_corrected
                #print('start Point {} \nEnd Point {} \nMidpoint {} \nDifference {} \nDistance {} \nva {} \nTravel time{}\n \n '.format(start_point, end_point, midpoint, difference, distance, va_corrected, traveltime ))
                time += traveltime
                continue
            n = self.densityfunctions.density_sep_equators(r, theta, phi)
            
            if n < 1e4: 
                n = 1e4
                ''' CURRENT LOW DENSITY CORRECTION'''

            #if r < 6:

            n_along_path.append(n)
            va = self.calculator(averageB, n)
            #print(averageB, n)
            va_uncorrected_list.append(va)

            va_corrected = self.relativistic_correction(va)
            va_corrected_list.append(va)
            
            
            traveltime = distance/va_corrected
            #print('start Point {} \nEnd Point {} \nMidpoint {} \nDifference {} \nDistance {} \nva {} \nTravel time{}\n \n '.format(start_point, end_point, midpoint, difference, distance, va_corrected, traveltime ))
            time += traveltime
        
        if print_time == 'on':
            print('travel time = {:.2f}s (= {:.1f}mins)'.format(time, time/60))
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
            x,y, z = self.plotter.make_sphere()
            ax.plot_surface(x, y, z, color = 'yellow', zorder=100, label = 'Jupiter')
            ax.set_xlim3d(-40, 40)
            ax.set_ylim3d(-40, 40)
            ax.set_zlim3d(-40, 40)
            ax.set_xlabel('$X, R_j$', fontsize=10)
            ax.set_ylabel('$Y, R_J$', fontsize=10)
            fig.suptitle('Field Line for which travel time was calculated using {}'.format(self.model))
            ax.set_title('Start Point = ({:.0f}, {:.1f}{}, {:.1f}{})SYSIII'.format(startpoint[0]/Rj, startpoint[1] * 180/np.pi, u"\N{DEGREE SIGN}", phi_lh * 180/np.pi, u"\N{DEGREE SIGN}"))
            ax.text2D(0.05, 0.95, 'time = {:.1f}s (= {:.1f}mins)'.format(time, time/60), transform=ax.transAxes)

            #plt.legend()
            plt.savefig('images-24-jan-update/travel_time_trace.png'.format(self.model))
            plt.show()
        if dr_plot == 'on':

            fig, ax1 = plt.subplots()
            numbers = list(range(len(drs_km)))
            
            ax1.plot(numbers, drs_km, label = 'Distance Between Points (km)', color = 'k')
            ax1.set_xlabel('Point Index')
            ax1.set_ylabel('Distance Between Points (km)', color = 'k')
            ax1.tick_params(axis='y', labelcolor='k')
            #ax1.legend(loc=0)

            ax2 = ax1.twinx() 
            ax2.plot(numbers, rs_km, label = 'r ($km$)', color = 'c', linestyle ='--')
            ax2.set_ylabel('Distance From Planet (km)', color = 'c')
            ax2.tick_params(axis='y', labelcolor='c')
            #ax2.legend(loc = 1)
            #plt.legend()
            plt.show()
        if va_plot == 'on':
            fig, ax1 = plt.subplots()
            numbers = list(range(len(va_corrected_list)))
            numbers_r = list(range(len(rs_rj_popped)))
            
            ax1.plot(numbers, va_corrected_list, label = '$v_A$ corrected ($ms^{-1}$)', color = 'c')
            ax1.set_xlabel('Point Index')
            ax1.set_ylabel('Speed ($ms^{-1}$)', color = 'b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.plot(numbers, va_uncorrected_list, label = '$v_A$ uncorrected ($ms^{-1}$)', color = 'b', linestyle ='--')
            #ax1.legend()
            ax2 = ax1.twinx() 
            ax2.plot(numbers_r, rs_rj_popped, label = 'Distance from planet ($R_J$)', color = 'r', linestyle ='-')
            ax2.set_ylabel('R ($R_J$)', color = 'r')
            ax2.tick_params(axis='y', labelcolor='r')
            #ax2.legend()
            #plt.grid(True)
            plt.figlegend()
            fig.suptitle('Effect of including relativistic correction')
            plt.savefig('images-24-jan-update/va correction effects.png')
            plt.show()
        if b_plot == 'on':
            fig, ax1 = plt.subplots()
            numbers_b = list(range(len(B_along_path)))
            
            ax1.plot(numbers_b, B_along_path, label = 'B along path', color = 'k')
            ax1.set_xlabel('Point Index')
            ax1.set_ylabel('B Along Path', color = 'k')
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
        if n_plot == 'on':
            fig, ax1 = plt.subplots()
            numbers_n = list(range(len(n_along_path)))
            
            ax1.plot(numbers_n, n_along_path, label = 'n along path', color = 'k')
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
            print(n_along_path[-1])
        return time, va_corrected_list, va_uncorrected_list, plottable_list, rs_km


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
            x,y, z = self.plotter.make_sphere()
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

    def multiple_travel_times(self, num = 8, plot = 'on', direction = 'forward'):
        ''' docstring goes here ''' 

        ''' TO START WITH, THIS IS JUST GONNA BE ALL ON ONE PLOT, BUT IT COULD BE EXTENDED TO HAVE THEM ALL ON SEPERATE PLOTS! ''' 
        startingPoints = []
        spacing = 2*np.pi/num
        for n in range(num):
            startingPoints.append([30, np.pi/2, n*spacing])
        if plot == 'on':
            fig = plt.figure()
            ax = fig.gca(projection='3d') # initialise figure
            colours = ['b','g','r','c','m','k',] # just setting this up for use later
            legend_elements = [] #MATPLOTLIB IS A PAIN
            #make the sphere 
            x,y, z = self.plotter.make_sphere()
            ax.plot_surface(x, y, z, color = 'yellow', zorder=100, label = 'Jupiter')
            ax.set_xlim3d(-40, 40)
            ax.set_ylim3d(-40, 40)
            ax.set_zlim3d(-40, 40)
            ax.set_xlabel('$X, R_j$', fontsize=10)
            ax.set_ylabel('$Y, R_J$', fontsize=10)
        color_index = 0
        angle_time_dictionary = {}
        for point in startingPoints:
            print('New Startpoint!')
            point[2] = 2*np.pi - point[2]
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

    def plot_angle_vs_time(self, num = 10, direction = "backward"):
        ''' generate a plot of how the travel time depends with the angle of the starting point. ''' 
        angles_times = self.multiple_travel_times(num=num, plot='off', direction=direction)
        #print(angles_times)
        angles = list(angles_times.keys())
        times = list(angles_times.values())
        angles_degree = [x*180/np.pi for x in angles]
        times_mins = [x/60 for x in times]
        #print(angles, times)
        fig, ax = plt.subplots()
        ax.plot(angles_degree, times_mins) 
        if direction == 'forward':
            endpoint = 'South'
        else:
            endpoint = 'North'
        ax.set(xlabel = 'phi $\u03BB_{III}$ (Degrees)', ylabel = 'Time (Minutes)', title ='Effect of Starting longitude In Equatorial Plane on Travel Time \n Destination: {} Hemsiphere'.format(endpoint))
        #ax.tick_params(labelright = True)
        ax.grid()
        plt.show()

    def plot_B_debug_time(self, phi_lh = 69):
        grids, gridz = self.help.makegrid_2d_negatives(200 ,gridsize= self.stop) #CHANGE THIS BACK TO 100 WHEN ITS WORKING
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
                B_r, B_theta, B_phi = self.field.Internal_Field(r, theta, phi, model=self.model) #calculates the magnetic field due to the internal field in spherical polar that point)
                B_current = self.field.CAN_sheet(r, theta, phi) #calculates the magnetic field due to the current sheet in spherical polar
                B_notcurrent = np.array([B_r, B_theta, B_phi]) 
                B_overall = np.add(B_current, B_notcurrent) #adds up the total magnetic field 
                B_x, B_y, B_z = self.help.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi)
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
        plt.savefig('images-24-jan-update/B side slice.png')
        plt.show() 

            


      


            


test = AlfvenVel(numpoints=200)
#test.top_down_matched_equators()

#test.topdown_seperate_equators(density = 'on')

#test.travel_time([30, np.pi/2, 69 * np.pi/180], direction='forward', dr_plot='off', path_plot = 'on', va_plot = 'off')
test.travel_time([30, np.pi/2, 50 * np.pi/180], direction='backward', dr_plot='off', path_plot = 'on', va_plot = 'on', b_plot='on', n_plot= 'on', debug_plot= 'on')
#test.sideview_seperate_equators(69)
#test.plot_rel_effect()
#test.plot_correction()
#test.multiple_travel_times(direction='backward')
#test.plot_B_debug_time()
#test.plot_angle_vs_time(num=100)