import math
import numpy as np
import sys
import matplotlib.pyplot as plt
import json
from helpful_functions import HelpfulFunctions
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, PathPatch
from matplotlib.colors import DivergingNorm
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
from copy import deepcopy
from field_and_current_sheet import InternalAndCS
plt.rcParams.update({'font.size': 22})
plt.rcParams['legend.fontsize'] = 14
Rj = 7.14 * (10 ** 7)
mu_0 = 1.25663706212 * 10 ** -6
personal_cmap = ['deeppink', 'magenta', 'darkmagenta' ,'darkorchid', 'indigo','midnightblue', 'darkblue', 'slateblue', 'dodgerblue', 'deepskyblue',  'aqua', 'aquamarine' ]
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
        ax.set_xlim(-self.stop,self.stop)
        ax.set_ylim(-self.stop,self.stop)
        degrees = theta * 180 /np.pi
        ax.set(xlabel = 'x $(R_J)$', ylabel = 'y $(R_J)$', title = 'alfven velocity in the colatitude = {:.0f}{} plane for aligned spin and centrifugal equators'.format(degrees, u"\N{DEGREE SIGN}"))
        fig.colorbar(cont, label = '$V_a (km)$')
        ax.set_aspect('equal', adjustable = 'box')
        for r in np.arange(0, 115, 5):
            ax.add_patch(Circle((0,0), r, fill = False, color = 'lightgreen'))
        plt.savefig('images-24-jan-update/va_topdown.png')
        #plt.show() 
        return Vas

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
        ax.set_xlim(-self.stop,self.stop)
        ax.set_ylim(-self.stop,self.stop)
        ax.set(xlabel = 'x $(R_J)$ ', ylabel = 'y $(R_J)$', title = 'alfven velocity in the spin plane, spin and centrifugal equators not aligned')
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
        return Vas

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
    
        
    def travel_time(self, startpoint = [30, np.pi/2, 212* np.pi/180], direction = 'forward', path_plot = 'off', dr_plot = 'off', print_time = 'on',
     va_plot ='off',b_plot = 'off', n_plot = 'off', vvsrplot = 'off', uncorrected_vvsr = 'off',
    debug_plot = 'off', nvsrplot = 'off', equators = 'unmatched'):
        '''
        Calculate the travel path/time of an alfven wave from a given startpoint to the ionosphere. 
        input startpoint [r, theta, phi] where r is in rj and phi is left handed in RADIANS
        direction input - 'forward' travels along the magnetic field vector to the southern hemisphere, backward is the reverse 

        ''' 
        saving_start = deepcopy(startpoint)
        startpoint[0] = startpoint[0]*Rj
        #startpoint[2] = 2*np.pi - startpoint[0]
        phi_lh = startpoint[2]

        #print('hello', startpoint, direction)
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
        
        firsttime = 0
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
            

            r, theta, phi = self.help.cart_to_sph(midpoint[0], midpoint[1], midpoint[2])
            r = r/Rj
            phi_lh = 2*np.pi - phi
            
            if r < 6: 
                if firsttime == 0:
                    if equators == 'unmatched':
                        n_at_6 =  self.densityfunctions.density_sep_equators(6, theta, phi_lh)
                    if equators == 'matched':
                        n_at_6 =  self.densityfunctions.density_same_equators(6, theta)
                    firsttime == 1
                '''
                this is where the density problem lies - we need to put a better version of the density in here! 
                ''' 
                
                n = self.densityfunctions.density_within_6(r, theta, phi_lh, n_at_6)
                n_along_path.append(n)
                va = self.calculator(averageB, n)
                va_uncorrected_list.append(va)
                va_corrected = self.relativistic_correction(va)
                va_corrected_list.append(va_corrected)
                traveltime = distance/va_corrected
                time += traveltime
                continue
            if equators =='unmatched':    
                n = self.densityfunctions.density_sep_equators(r, theta, phi_lh)
            if equators =='matched':    
                n = self.densityfunctions.density_same_equators(r, theta)
            #print(n)
            
            if n < 1e4: 
                n = 1e4
                ''' CURRENT LOW DENSITY CORRECTION'''
            n_along_path.append(n)
            va = self.calculator(averageB, n)
            #print(averageB, n)
            va_uncorrected_list.append(va)

            va_corrected = self.relativistic_correction(va)
            va_corrected_list.append(va_corrected)
            
            
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
            #ax.text(1,2,40, 'time = {:.1f}s (= {:.1f}mins)'.format(time, time/60))
            ax.text2D(0.05, 0.95, 'time = {:.1f}s (= {:.1f}mins)'.format(time, time/60), transform=ax.transAxes)

            #plt.legend()
            plt.savefig('images-24-jan-update/travel_time_trace.png'.format(self.model))
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
            ax2.plot(numbers, rs_rj, label = 'r ($R_j$)', color = 'c', linestyle ='--')
            ax2.set_ylabel('Distance From Planet ($R_j$)', color = 'c')
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
            ax2.set_ylabel('R ($R_J$)', color = 'r')
            ax2.tick_params(axis='y', labelcolor='r')
            #ax2.legend()
            #plt.grid(True)
            plt.figlegend()
            fig.suptitle('Effect of including relativistic correction')
            plt.savefig('images-24-jan-update/va correction effects.png')
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
            ax2.plot(numbers_r, rs_rj, label = 'r ($R_j$)', color = 'c', linestyle ='--')
            ax2.set_ylabel('Distance From Planet ($R_j$)', color = 'c')
            ax2.tick_params(axis='y', labelcolor='c')
            #ax2.legend(loc = 1)
            #plt.legend()
            plt.grid()
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
            ax1.plot(rs_rj[:-1], va_corrected_list, label = '$v_A$ corrected ($ms^{-1}$)', color = 'c')
            ax1.set_xlabel('R $(R_J)$')
            ax1.set_ylabel('Alfven Velocity ($ms^{-1}$)', color = 'b')
            ax1.plot(rs_rj[:-1], va_uncorrected_list, label = '$v_A$ uncorrected ($ms^{-1}$)', color = 'b', linestyle ='--')

            ax1.legend()
            ax1.set_title('Alfven Velocity Against Distance from planet \n Including effect of including relativistic correction')
            plt.savefig('images-24-jan-update/va vs r.png')
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
            plt.savefig('images-24-jan-update/va vs r.png')
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

    def multiple_travel_times(self, num = 8, plot = 'on', direction = 'forward', r = 30):
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

    def plot_angle_vs_time(self, num = 10, direction = "backward", r = 30):
        ''' generate a plot of how the travel time depends with the angle of the starting point. ''' 
        angles_times = self.multiple_travel_times(num=num, plot='off', direction=direction, r=r)
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
        ax.set(xlabel = 'phi $\u03BB_{III}$ (Degrees)', ylabel = 'Time (Minutes)', 
        title ='Effect of Starting longitude In Equatorial Plane on Travel Time \n From r = {}$R_J$ to Destination: {} Hemsiphere'.format(r,endpoint))
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

    def multiple_travel_times_both_directions(self, num = 8, r = 10, equators = 'unmatched'):
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
            calc_f = self.travel_time(startpoint=point, print_time='on', direction = 'forward', equators=equators)
            print('point after calc f', point)
            point[0] = point[0]/Rj
            point[2] = 2*np.pi - point[2]
            print('amended point', point)
            #print('got here, calc_f 0 =', calc_f[0], ' point = ' ,point)
            calc_b = self.travel_time(startpoint=point, print_time='on', direction = 'backward', equators=equators)
            time_f = calc_f[0]
            time_b = calc_b[0]
            time = time_b + time_f
            angle_time_dictionary[phi_lh] = time
        return angle_time_dictionary        

    def plot_angle_vs_time_btoh_directions(self, num = 10, r = 10, equators = 'unmatched'):
        ''' generate a plot of how the travel time depends with the angle of the starting point. ''' 
        angles_times = self.multiple_travel_times_both_directions(num=num, r=r, equators=equators)
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

    def plot_multiple_distances_both_directions(self, num = 50, r = 10, equators = 'unmatched'):
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
            calc_f = self.travel_time(startpoint=point, print_time='on', direction = 'forward', equators=equators)
            print('point after calc f', point)
            point[0] = point[0]/Rj
            point[2] = 2*np.pi - point[2]
            print('amended point', point)
            #print('got here, calc_f 0 =', calc_f[0], ' point = ' ,point)
            calc_b = self.travel_time(startpoint=point, print_time='on', direction = 'backward', equators=equators)
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

    def plot_radial_outflow_countour(self, mdot, gridsize = 40):
        gridx, gridy = self.help.makegrid_2d_negatives(200 ,gridsize= gridsize)
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
                flow_vel = self.radialfunctions.flow_velocity(r, mdot)
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
        plt.savefig('images-24-jan-update/v outflow equatorial')
        plt.show() 
        return vOutflows

    def plot_outflow_vs_alfven(self, mdot, equators = 'matched', gridsize = 20, model = 'VIP4', cansheet = 'off'):
        ''' 
        equators = 'matched' - spin and centrifgual axis are the same
                 = 'unmatched' - spin and centrfigugal equators matched. 

        '''
        theta = np.pi/2
        gridx, gridy = self.help.makegrid_2d_negatives(400 ,gridsize= gridsize)
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
                flow_vel = self.radialfunctions.flow_velocity(r, mdot)
                if equators == 'matched':
                    n = self.densityfunctions.density_same_equators(r, theta)
                phi = np.arctan2(y,x) 
                if equators == 'unmatched':
                    n = self.densityfunctions.density_sep_equators(r, theta, phi)
                
                B_r, B_theta, B_phi = self.field.Internal_Field(r, theta, phi, model=model) #calculates the magnetic field due to the internal field in spherical polar that point)
                if cansheet == 'on':
                    B_current = self.field.CAN_sheet(r, theta, phi) #calculates the magnetic field due to the current sheet in spherical polar
                else: 
                    B_current = [0,0,0]
                B_notcurrent = np.array([B_r, B_theta, B_phi]) 
                B_overall = np.add(B_current, B_notcurrent) #adds up the total magnetic field 
                B_x, B_y, B_z = self.help.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi)
                B = np.array([B_x, B_y, B_z])
                B = B/(1e9) #CHRIS code outputs nT
                '''
                ISSUE MAY BE HERE AS THE MANGITUDE OF B IS NOT TAKEN? - nope the calculator expects this.
                '''
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
        levs = [0,0.25,0.5,0.75,0.8,0.9,1,1.1,1.2,1.25,1.50,1.75,2]#, np.max(va_over_outflow)]
        plt.gca().patch.set_color('.25')
        cont = ax.contourf(gridx, gridy, va_over_outflow, cmap = 'seismic',norm = DivergingNorm(vcenter = 1), levels = levs)#levels = levs, # norm=mcolors.LogNorm())# locator=ticker.LogLocator())
        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        ax.add_patch(Circle((0,0), 6.2, color='c', zorder=90, label = "Io Orbital Radius"))
        ax.set_xlim(-gridsize,gridsize)
        ax.set_ylim(-gridsize,gridsize)
        for r in np.arange(0, 115, 5):
            ax.add_patch(Circle((0,0), r, fill = False, color = 'lightgreen'))
        ax.set(xlabel = 'X $(R_J)$ \n', ylabel = 'Y $(R_J)$', title = 'Radial Outflow vs Alfven Velocity in Equatorial Plane \n mdot = {}, equators = {}'.format(mdot, equators))
        fig.colorbar(cont, label = ' Alfven Velocity / Radial Velocity', ticks = levs)
        ax.set_aspect('equal', adjustable = 'box')
        ax.legend()
        plt.savefig('images-24-jan-update/v outflow vs VA equatorial')
        plt.show()

    def diverge_rel_correction(self, startpoint = [10, np.pi/2, 200.8* np.pi/180], direction = 'forward', rtol = 0.01):
        calc = self.travel_time(startpoint=startpoint, direction=direction)
        points = calc[5]
        corrected = calc[1]
        uncorrected = calc[2]
        diverge_index = self.find_index_diverge(corrected, uncorrected, rtol=rtol)
        
        diverge_point = points[diverge_index]
        diverge_point_sph = self.help.cart_to_sph(diverge_point[0], diverge_point[1], diverge_point[2])
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
    
    def relativistic_correction_area_of_impact_2d(self, phi_lh_deg, equators = 'matched' ,rtol = 0.01, numpoints = 200):
        ''' 
        plot the areas in which the relativistic correction has an impact on the alfven velocity
        Inputs:
        phi_lh_deg - the longitude you want to view in left handed sysIII, in degrees
        equators - input either "matched" or "unmatched". if equators = "matched", then spin and centrifugal equators are aligned. If "unmatched" centrifugal equator is 
                   seperately calculated
        rtol = the relative tolerance between uncorrected and corrected alfven velocity to define where the correction has an impact, default 1%
        '''
        
        phi_lh_rad = phi_lh_deg * np.pi/180
        
        ''' first define grid of points at which the velocities will be calculated ''' 
        grids, gridz = self.help.makegrid_2d_negatives(numpoints ,gridsize= 30)
        
        
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
                    if firsttime == 0:
                        if equators == 'unmatched':
                            n_at_6 =  self.densityfunctions.density_sep_equators(6, theta, phi)
                        if equators == 'matched':
                            n_at_6 =  self.densityfunctions.density_same_equators(6, theta)
                        firsttime == 1
                    
                    n = self.densityfunctions.density_within_6(r, theta, phi, n_at_6) #in order to change the density profile within 6rj, this is what should be changed. 
                    ''' use chris' mag field models code to add the magnetic field from the internally generated field and the current sheet ''' 
                    B_r, B_theta, B_phi = self.field.Internal_Field(r, theta, phi, model=self.model) #calculates the magnetic field due to the internal field in spherical polar that point)
                    B_current = self.field.CAN_sheet(r, theta, phi) #calculates the magnetic field due to the current sheet in spherical polar
                    B_notcurrent = np.array([B_r, B_theta, B_phi]) 
                    B_overall = np.add(B_current, B_notcurrent)
                    B_x, B_y, B_z = self.help.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi)
                    
                    B = np.array([B_x, B_y, B_z])
                    B_tesla = B/(1e9) #CHRIS code outputs nT
                    va = self.calculator(B_tesla, n)
                    va_uncorrected_row.append(va)
                    va_corrected = self.relativistic_correction(va)
                    va_corrected_row.append(va_corrected)
                    continue #after its calculated for r<6 back to the top of the loop
            
                ''' if r>6, then density and mangetic field calculated in the same manner ''' 
                if equators =='unmatched':    
                    n = self.densityfunctions.density_sep_equators(r, theta, phi_lh)
                if equators =='matched':    
                    n = self.densityfunctions.density_same_equators(r, theta)
                #print(n)

                B_r, B_theta, B_phi = self.field.Internal_Field(r, theta, phi, model=self.model) #calculates the magnetic field due to the internal field in spherical polar that point)
                B_current = self.field.CAN_sheet(r, theta, phi) #calculates the magnetic field due to the current sheet in spherical polar
                B_notcurrent = np.array([B_r, B_theta, B_phi]) 
                B_overall = np.add(B_current, B_notcurrent)
                B_x, B_y, B_z = self.help.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi)
                
                B = np.array([B_x, B_y, B_z])
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
        plt.savefig('images-24-jan-update/where va correction matterss 2d')
        plt.show()
        '''
        ''' plot v2 ''' 
        divided = np.array(va_corrected_list)/np.array(va_uncorrected)
        fig, ax = plt.subplots()
        plt.gca().patch.set_color('.25')
        #print(are_close)

        cont = ax.contourf(grids, gridz, divided, levels = [0,0.5,0.6,0.7,0.8,0.9,0.95,0.96,0.97,0.98,0.99,1], colors = personal_cmap)#levels = [0.2,0.9])#levels = levs, # norm=mcolors.LogNorm())# locator=ticker.LogLocator())

        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        ax.add_patch(Circle((0,0), 6.2, color='c', zorder=90, label = "Io Orbital Radius", fill = False))
        ax.set_xlim(-30,30)
        ax.set_ylim(-30,30)
        for r in np.arange(0, 45, 5):
            ax.add_patch(Circle((0,0), r, fill = False, color = 'lightgreen'))
        ax.set(xlabel = ' $(R_J)$ \n', ylabel = ' $(R_J)$', title = 'Corrected/Uncorrected Alfven velocity in phi = {:.0f} plane \n  equators = {}'.format(phi_lh_deg, equators, rtol))
        fig.colorbar(cont) 
        ax.set_aspect('equal', adjustable = 'box')
        ax.legend()
        plt.savefig('images-24-jan-update/where va correction matters 2d v2')
        plt.show()

    def relativistic_correction_area_of_impact_topdown(self,equators = 'matched'):
        ''' 
        equators = 'matched' - spin and centrifgual axis are the same
                 = 'unmatched' - spin and centrfigugal equators matched. 

        '''
        theta = np.pi/2
        gridx, gridy = self.help.makegrid_2d_negatives(200 ,gridsize= 30)
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
                    if firsttime == 0:
                        if equators == 'unmatched':
                            n_at_6 =  self.densityfunctions.density_sep_equators(6, theta, phi)
                        if equators == 'matched':
                            n_at_6 =  self.densityfunctions.density_same_equators(6, theta)
                        firsttime == 1
                    
                    n = self.densityfunctions.density_within_6(r, theta, phi, n_at_6) #in order to change the density profile within 6rj, this is what should be changed. 
                    ''' use chris' mag field models code to add the magnetic field from the internally generated field and the current sheet ''' 
                    B_r, B_theta, B_phi = self.field.Internal_Field(r, theta, phi, model=self.model) #calculates the magnetic field due to the internal field in spherical polar that point)
                    B_current = self.field.CAN_sheet(r, theta, phi) #calculates the magnetic field due to the current sheet in spherical polar
                    B_notcurrent = np.array([B_r, B_theta, B_phi]) 
                    B_overall = np.add(B_current, B_notcurrent)
                    B_x, B_y, B_z = self.help.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi)
                    B = np.array([B_x, B_y, B_z])
                    B_tesla = B/(1e9) #CHRIS code outputs nT
                    va = self.calculator(B_tesla, n)
                    va_uncorrected_row.append(va)
                    va_corrected = self.relativistic_correction(va)
                    va_corrected_row.append(va_corrected)
                    divided = va_corrected/va
                    va_corrected_over_uncorrected_row.append(divided)
                    continue

                if equators == 'matched':
                    n = self.densityfunctions.density_same_equators(r, theta)
                 
                if equators == 'unmatched':
                    n = self.densityfunctions.density_sep_equators(r, theta, phi)
                
                B_r, B_theta, B_phi = self.field.Internal_Field(r, theta, phi, model='VIP4') #calculates the magnetic field due to the internal field in spherical polar that point)
                B_current = self.field.CAN_sheet(r, theta, phi) #calculates the magnetic field due to the current sheet in spherical polar
                B_notcurrent = np.array([B_r, B_theta, B_phi]) 
                B_overall = np.add(B_current, B_notcurrent) #adds up the total magnetic field 
                B_x, B_y, B_z = self.help.Bsph_to_Bcart(B_overall[0], B_overall[1], B_overall[2], r, theta, phi)
                B = np.array([B_x, B_y, B_z])
                B = B/(1e9) #CHRIS code outputs nT
                va = self.calculator(B, n)
                va_uncorrected_row.append(va)
                va_corrected = self.relativistic_correction(va)
                va_corrected_row.append(va_corrected)
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
        ax.set(xlabel = ' $(R_J)$ \n', ylabel = ' $(R_J)$', title = 'Corrected/Uncorrected Alfven velocity in equatorial plane \n  equators = {}'.format(equators))
        fig.colorbar(cont) 
        ax.set_aspect('equal', adjustable = 'box')
        ax.legend()
        plt.savefig('images-24-jan-update/where va correction matters 2d topdown')
        plt.show()

            
        #outflows_km= np.array(vOutflows)/1e3
        #va_over_outflow = [va/vo for va,vo in zip(vas,vOutflows)]
        #va_over_outflow = []
        
        

        fig, ax = plt.subplots(figsize = (25,16))
        #lev_exp = np.arange(np.floor(np.log10(np.min(va_over_outflow))-1), np.ceil(np.log10(np.max(va_over_outflow))+1), step = 0.25)
        #levs = np.power(10, lev_exp)
        

    def outflow_vs_alfven_v2(self,  mdot, equators = 'matched', gridsize = 80, model = 'VIP4'):
        outflows = np.array(self.plot_radial_outflow_countour(mdot= mdot))
        if equators == 'matched':
            alfven = np.array(self.top_down_matched_equators())
        if equators == 'unmatched':
            alfven = np.array(self.spin_eq_topdown())
        gridx, gridy = self.help.makegrid_2d_negatives(200 ,gridsize= gridsize)
        divided = np.divide(alfven, outflows)
        print(divided)
        #np.savetxt('divided.txt', divided)
        fig, ax = plt.subplots(figsize = (25,16))
        levs = [0,0.25,0.5,0.75,0.8,0.9,1,1.1,1.2,1.25,1.50,1.75,2,5]#, np.max(va_over_outflow)]
        plt.gca().patch.set_color('.25')
        cont = ax.contourf(gridx, gridy, divided, cmap = 'seismic',norm = DivergingNorm(vcenter = 1), levels = levs)#levels = levs, # norm=mcolors.LogNorm())# locator=ticker.LogLocator())
        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        ax.add_patch(Circle((0,0), 6.2, color='c', zorder=90, label = "Io Orbital Radius"))
        ax.set_xlim(-gridsize,gridsize)
        ax.set_ylim(-gridsize,gridsize)
        for r in np.arange(0, 115, 5):
            ax.add_patch(Circle((0,0), r, fill = False, color = 'lightgreen'))
        ax.set(xlabel = 'X $(R_J)$ \n', ylabel = 'Y $(R_J)$', title = 'Radial Outflow vs Alfven Velocity in Equatorial Plane \n mdot = {}, equators = {}'.format(mdot, equators))
        fig.colorbar(cont, label = ' Alfven Velocity / Radial Velocity', ticks = levs)
        ax.set_aspect('equal', adjustable = 'box')
        ax.legend()
        plt.savefig('images-24-jan-update/v outflow vs VA equatorial v2')
        plt.show()




    #def relativistic_correction_area_of_impact_3d(self)

test = AlfvenVel(numpoints=200, model='VIP4', stop = 80)
#test.top_down_matched_equators()
#test.topdown_seperate_equators(density = 'on')
#test.top_down_matched_equators()
#test.plot_radial_outflow_countour(mdot =500)
#test.plot_outflow_vs_alfven(mdot = 500, equators='matched')
#test.travel_time([30, np.pi/2, 69 * np.pi/180], direction='forward', dr_plot='off', path_plot = 'on', va_plot = 'off')
#test.travel_time([10, np.pi/2, 200.8 * np.pi/180], direction='forward', dr_plot='off', path_plot = 'off', va_plot = 'off', b_plot='off', n_plot= 'off', 
#debug_plot= 'off', nvsrplot = 'on', vvsrplot = 'on', uncorrected_vvsr = 'on')
#test.sideview_seperate_equators(111)
#test.plot_rel_effect()
#test.plot_correction()
#test.multiple_travel_times(direction='backward')
#test.plot_B_debug_time()
#test.plot_angle_vs_time(num=100, r = 14)
#test.plot_multiple_distances(num = 70)
#test.plot_angle_vs_time_btoh_directions(r=10, num=70, equators = 'matched')
#test.plot_multiple_distances_both_directions(num=50, r=10, equators='matched')
#test.plot_outflow_vs_alfven(mdot = 500, gridsize = 80, model='VIP4', cansheet = 'on', equators = 'matched')
#test.diverge_rel_correction()
#test.visualise_rel_correction_single_point()
#test.relativistic_correction_area_of_impact_2d(200.8)
#test.relativistic_correction_area_of_impact_topdown()
test.outflow_vs_alfven_v2(mdot = 500)