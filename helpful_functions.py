'''
There seems to be functions that I keep repeating, and instead of writing the same thing out multiple times it makes more sense just to have them all here.

This will hopefully be added to as we go! 
'''
import numpy as np 
import math
class HelpfulFunctions():

    def __init__(self):
        pass 

    def makegrid(self,NumPoints, gridsize):
        points_min = -gridsize/2
        points_max = gridsize/2
        x,y,z = np.meshgrid(np.linspace(points_min, points_max, NumPoints),
                            np.linspace(points_min, points_max, NumPoints),
                            np.linspace(points_min, points_max, NumPoints))
        return x,y,z

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
'''
hpf = HelpfulFunctions()
test2_point = [0,1,0]
'''