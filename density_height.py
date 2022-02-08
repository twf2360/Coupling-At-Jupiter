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
import matplotlib.colors as mcolors
from radial_outflow import radialOutflow
from matplotlib import ticker, cm
from field_and_current_sheet import InternalAndCS
Rj = 7.14 * 10 ** 7
plt.rcParams.update({'font.size': 22})
plt.rcParams['legend.fontsize'] = 14
class DensityHeight:
    def __init__(self, numpoints, start, stop):
        self.radialOutflowFunctions = radialOutflow(28)
        self.numpoints = numpoints
        self.help = HelpfulFunctions()
        self.start = start
        self.stop = stop 
        self.tracing = InternalAndCS()
    
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

    def density(self, n_0, z, H):
        n = n_0 * np.exp(-z/H)
        return n

    def plotting(self, density = 'on',scale_height = 'off'):
        radii, n_0s = self.radialOutflowFunctions.plotRadialDensity(start=self.start, end = self.stop, numpoints=self.numpoints)
        zs =  np.linspace(self.start, self.stop, self.numpoints)
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
            for r in radii:
                Hs.append(self.scaleheight(r))
            fig, ax = plt.subplots(figsize =(25,13))
            ax.plot(radii, Hs, label = 'Scale Height', color = 'g')
            #plt.xscale('log')
            ax.set(xlabel='Radius (RJ)', ylabel='Scale Height ($R_J$)', title='Scale height depenence on radial distance')
            plt.xlim(0, 100)
            plt.show()
    
    def equators_line(self):
        '''
        this does technically plot the equators, but the centrifugal equator as plotted as a straight line, which it just isn't. 
        '''
        r = 30
        theta = np.pi/2
        phi_LH =  21* np.pi/180
        phi_rh = 2 *np.pi - phi_LH
        Btheta_eq = self.tracing.find_mag_equator(point=[r*Rj, theta,phi_LH])

        centrifugaleq_theta_lat = self.help.centrifugal_equator(r, phi_rh)# +np.pi)
        print(Btheta_eq, centrifugaleq_theta_lat)

        centrifual_eq = np.array([[-r * np.cos(centrifugaleq_theta_lat), - r * np.sin(centrifugaleq_theta_lat)], [r * np.cos(centrifugaleq_theta_lat),  r * np.sin(centrifugaleq_theta_lat)]])
        b_eq = np.array([[-r * np.sin(Btheta_eq), - r * np.cos(Btheta_eq)], [r * np.sin(Btheta_eq),  r * np.cos(Btheta_eq)]])
        spin_eq = np.array([[-r,0],[r,0]])

        centrifual_eq_t = np.transpose(centrifual_eq)
        b_eq_t = np.transpose(b_eq)
        spin_eq_t = np.transpose(spin_eq)

        fig, ax = plt.subplots()
        
        ax.plot(centrifual_eq_t[0] , centrifual_eq_t[1], color = 'c', label = 'Centrifugal Equator')
        ax.plot(spin_eq_t[0], spin_eq_t[1], color = 'm', label = 'Spin Equator')
        ax.plot(b_eq_t[0], b_eq_t[1], color = 'k', label = 'Magnetic Field Equator')

        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        phi_lh_deg = phi_LH * 180/np.pi 
        ax.set(xlim = (-30,30), ylim = (-5,5), xlabel = 'X $R_J$', ylabel = 'Y $R_J$', title = 'Different Equators at Jupiter, SYSIII longitude =   {:.0f} deg on RHS'.format(phi_lh_deg))
        ax.set_aspect(aspect='equal')
        plt.legend()
        plt.show()

    def equators_cent_calculated(self):
        r = 30
        theta = np.pi/2
        phi_LH =  21* np.pi/180
        phi_rh = 2 *np.pi - phi_LH
        Btheta_eq = self.tracing.find_mag_equator(point=[r*Rj, theta,phi_LH])
        b_eq = np.array([[-r * np.sin(Btheta_eq), - r * np.cos(Btheta_eq)], [r * np.sin(Btheta_eq),  r * np.cos(Btheta_eq)]])
        spin_eq = np.array([[-r,0],[r,0]])
        b_eq_t = np.transpose(b_eq)
        spin_eq_t = np.transpose(spin_eq)

        numpoints = 100
        centrifugal_equator = []
        cent_points = np.linspace(-r, r, num=numpoints)
        for x_point in cent_points:
            H = self.help.height_centrifugal_equator(x_point, phi_rh)
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
        plt.savefig('images/equators.png')
        plt.show()

    def meridian_slice(self, phi_lh, lines = 'off', num = 200):
        ''' 
        plots a slice of the density at a certain longitude given by phi_lh (in degrees)
        '''
        phi_lh_rad = phi_lh * np.pi/180
        phi_rh = 2*np.pi - phi_lh_rad
        
        #print(phi_rh)
        densities = []
        grids, gridz = self.help.makegrid_2d_negatives(200 ,gridsize= self.stop)

        r_cent_points = np.linspace(-30, 30, num=num)
        cent_plot_points = []
        mag_plot_points = []
        r_centtheta_magtheta_dict = {}

        for point in r_cent_points:
            if point > 0:
                phi = phi_rh + np.pi 
                phi_lh_for_calc = phi_lh_rad + np.pi
            else: 
                phi = phi_rh 
                phi_lh_for_calc = phi_lh_rad
            if -1 < point <1: 
                continue 

            #r_mag, theta_mag_S, phi_mag = self.help.simple_mag_equator(abs(point), np.pi/2, phi_lh_for_calc)
            theta_mag_colat = self.help.complex_mag_equator(abs(point),  phi_lh_for_calc)
            #print(theta_mag_colat, theta_mag_S)
            #print(point, phi_lh_for_calc)
            #print(point, theta_mag_colat,  (np.pi/2) - theta_mag_colat )
            theta_mag = np.pi/2 - (theta_mag_colat -np.pi/2)
            #print(theta_mag_colat,theta_mag)
            r_cent, theta_cent, phi_cent = self.help.change_equators(abs(point), np.pi/2, phi)
            r_centtheta_magtheta_dict[point] = [theta_cent, theta_mag]
            
            z_cent = abs(point) * np.cos(theta_cent)
            z_mag = abs(point) * np.cos(theta_mag)
            #print(point,theta_cent, theta_mag)

            mag_plot_points.append([point, z_mag]) 
            cent_plot_points.append([point, z_cent]) 
        cent_plot_points = np.array(cent_plot_points)
        mag_plot_points = np.array(mag_plot_points)

        mag_plot_points_t = np.transpose(mag_plot_points)
        cent_plot_points_t = np.transpose(cent_plot_points)
        spin_eq_plot = np.array([[-30,0], [30,0]])
        spin_eq_plot_t = np.transpose(spin_eq_plot)
        
        for i in range(len(gridz)):
            #print('new row, {} to go'.format(len(gridz)-i))
            
            density_row = []
            for j in range(len(grids)):
                z = gridz[i][j]
                s = grids[i][j]
                r = np.sqrt(z**2 + s**2)
                phi = phi_rh 

                if r < 6:
                    n = 1e6
                    ''' obviously this will have to change at somepoint '''
                    density_row.append(n)
                    continue
                theta = np.arctan2(s,z)
                n = self.density_sep_equators(r, theta, phi)
                density_row.append(n)
            densities.append(density_row)
        
        densities_cm = np.array(densities)/1e6
        densities_cm_edits = np.clip(densities_cm, 1e-2, 1e10)
        fig, ax = plt.subplots(figsize = (25,16))
        lev_exp = np.arange(np.floor(np.log10(densities_cm_edits.min())-1), np.ceil(np.log10(densities_cm_edits.max())+1), step = 0.25)
        levs = np.power(10, lev_exp)
        cont = ax.contourf(grids, gridz, densities_cm_edits, cmap = 'bone', levels = levs, norm=mcolors.LogNorm())#, locator=ticker.LogLocator()) #, levels = 14)
        ax.add_patch(Circle((0,0), 1, color='y', zorder=100, label = "Jupiter"))
        ax.add_patch(Circle((0,0), 6, color='c', zorder=90, label = "Io Orbital Radius"))
        ax.text(0.95, 0.01, 'SYS III (LH) Longitutude = {}{} '.format(phi_lh, u"\N{DEGREE SIGN}"),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='w', fontsize=16)
        if phi_lh + 180 > 360:
            text_degree = phi_lh - 180
        else:
            text_degree = phi_lh + 180
        ax.text(0.05, 0.99, 'SYS III (LH) Longitutude = {}{} '.format(text_degree, u"\N{DEGREE SIGN}"),
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes,
        color='w', fontsize=16)
        ax.text(0.05, 0.05, 'CML 201 $\u03BB_{III}$',
        verticalalignment='top', horizontalalignment='left',
        transform=ax.transAxes,
        color='w', fontsize=16)
        

        ax.set_xlim(-30,30)
        ax.set_ylim(-15,15)
        ax.set(xlabel = ' x($R_J$) \n', ylabel = 'y ($R_J$)', title = 'Density Contour Plot for Given longitude') #, title = 'CML 202 $\u03BB_{III}$')
        if lines == 'on':
            ax.plot(mag_plot_points_t[0], mag_plot_points_t[1], label = 'Magnetic Equator', color = 'm')
            ax.plot(cent_plot_points_t[0], cent_plot_points_t[1], label = 'Centrifugal Equator')
            ax.plot(spin_eq_plot_t[0], spin_eq_plot_t[1], label = 'Spin Equator')
        fig.colorbar(cont, label = 'Density $(cm^{-3})$')
        ax.set_aspect('equal', adjustable = 'box')
        ax.legend()
        plt.savefig('images-24-jan-update/density longitude slice.png')
        plt.show() 
        return r_centtheta_magtheta_dict
        


                
                 


    def density_sep_equators(self, r, theta, phi):
        ''' 
        Input r, theta (colatitude), phi (rh), in right handed sysII co-ordiantes
        returns mass density, taking into account the difference between spin and centrifugal equators. 
        '''
        r_cent = r 
        phi_cent = phi
        theta_shift = self.help.centrifugal_equator(r, phi)
        theta_cent = theta + theta_shift
        scaleheight = self.scaleheight(r_cent)
        n_0 = self.radialOutflowFunctions.radial_density(r_cent)
        z_cent =  r_cent * np.cos(theta_cent)
        n = self.density(n_0, abs(z_cent), scaleheight)

        return n 


    def equator_comparison_mag_cent(self, phi = 200, num = 200):
        r_thetas_dict = self.meridian_slice(phi_lh = phi, num= num)
        r_thetas_dict_positive_only = {k: v for (k, v) in r_thetas_dict.items() if k >= 6}
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





test = DensityHeight(numpoints= 100, start= 5, stop = 30)
#test.plotting(scale_height='on', density = 'on')    
#test.equators_cent_calculated()
#test.density_sep_equators(30, np.pi/2, 360*np.pi/180)
#test.meridian_slice(20.8, lines='on')

test.equator_comparison_mag_cent(20.8, num=500)
