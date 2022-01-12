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
        points_min = -gridsize/2
        points_max = gridsize/2
        x,y,z = np.meshgrid(np.linspace(points_min, points_max, NumPoints),
                            np.linspace(points_min, points_max, NumPoints),
                            np.linspace(points_min, points_max, NumPoints))
        return x,y,z
    
    def makegrid_2d(self, numpoints, gridsize):
        x,y = np.meshgrid(np.linspace(0, gridsize, numpoints),
                            np.linspace(0, gridsize, numpoints))
        return x,y

    def makegrid_2d_negatives(self, numpoints, gridsize):
        x,y = np.meshgrid(np.linspace(-gridsize, gridsize, numpoints *2),
                            np.linspace(-gridsize, gridsize, numpoints*2))
        return x,y
        
    def sph_to_cart(self, r, theta, phi):
        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)
        return x,y,z



    def cart_to_sph(self,x ,y,z):
        r = np.sqrt(x**2+y**2+z**2)
        theta = np.arctan2(np.sqrt(x**2+y**2),z)
        phi = np.arctan2(y,x)
        return r, theta, phi

    def Bsph_to_Bcart(self, Br, Btheta, Bphi, r, theta,phi):
        theta = float(theta)
        phi = float(phi)
        Bx = Br * np.sin(theta) * np.cos(phi) + Btheta * np.cos(theta) * np.cos(phi) - Bphi *np.sin(phi)
        By = Br * np.sin(theta) * np.sin(phi) + Btheta * np.cos(theta) * np.sin(phi) + Bphi *np.cos(phi)
        Bz = Br * np.cos(theta)  - Btheta * np.sin(theta)
        return Bx, By, Bz



    def unit_vector_cart(self, vector):
        norm = np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
        return vector/norm

    def centrifugal_equator(self, r, phi):
        ''' 
        R SHOULD BE INPUT IN RJ
        PHI SHOULD BE IN WEST (RH) LONGITUDE
        eq from https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2020JA028713 eq 2 
        returns the height over the jovigraphic equator
        '''
        phiLH = 2 * np.pi - phi
        a = 1.66 * np.pi / 180
        b = 0.131
        c = 1.62
        d = 7.76 * np.pi /180
        e = 249 * np.pi/180
        centrifualEq = (a * np.tanh(b*r -c)+ d) * np.sin(phiLH - e)

        return centrifualEq 

        

