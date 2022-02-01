'''
There seems to be functions that I keep repeating, and instead of writing the same thing out multiple times it makes more sense just to have them all here.

This will hopefully be added to as we go! 
'''
import numpy as np 
import math
import sys
class HelpfulFunctions():

    def __init__(self):
        pass 

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
        phiLH = 2 * np.pi - phi
        a = 1.66 * np.pi / 180
        b = 0.131
        c = 1.62
        d = 7.76 * np.pi /180
        e = 249 * np.pi/180
        centrifualEq = (a * np.tanh(b*r -c)+ d) * np.sin(phiLH - e)
        

        return centrifualEq 

    def change_equators(self, r, theta, phi):
        r_cent = r 
        phi_cent = phi
        theta_shift = self.centrifugal_equator(r, phi)
        theta_cent = theta + theta_shift
        return r, theta_cent, phi_cent
        
    def change_equators_cart_output(self, r, theta, phi):
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

    def simple_mag_equator(self, r, theta, phi_lh):
        ''' r in rj, theta colatitude, phi lh in rad''' 
        r = r
        phi = phi_lh
        tilt_direction = 159.2 * np.pi/180 #200.8 seems to be right. but 159.2 
        tilt_magnitude = 9.6 * np.pi/180
        tilt = tilt_magnitude * np.cos(phi_lh - tilt_direction)
        theta = theta - tilt
        return r, theta, phi
        '''so jupiters mag field is tilted by apporximated 9.6 deg using vip4 towards 200.8 lh longitude '''
    '''
    #################################### NOTE ######################################
    THE TWO BELOW FUNCTIONS SHOULDN'T BE USED - i don't think they are effective, however i am loathe to delete them jic
    '''
    
    def height_centrifugal_equator(self, r, phi):
        ''' 
        Input the r, phi coordinates of a point on the spin equator of Jupiter
        Note: R is in Rj, Phi is right handed 
        Returns H, the height above centrifugal equator
        '''        
        latitude = self.centrifugal_equator(r, phi)
        H = r * np.sin(latitude)
        return H 
    
    def length_centrifual_equator(self, r, phi):
        ''' 
        as r_spin and r_cent are not the same.
        Input r_spin, returns d (=r_cent)
        '''
        latitude = self.centrifugal_equator(r, phi)
        d = r * np.cos(latitude)
        return d


        

