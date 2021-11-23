''' 
this is just so that i can check that the two magnetic fields look normal when added together, so it doesn't need the 
same kind of adabptability etc. as the other programs 
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
Rj = 7.14e7
class current_trace:

    def __init__(self, numPoints):
        self.field = field_models()
        self.numPoints = numPoints
        self.helpful = HelpfulFunctions()

        

    def trace(self):
        coordinates = [30* Rj, 0, 0]
        pointslower = []
        pointsupper = []
        i, j = 0,0
        print('\n lower hemisphere')
        for i in range(self.numPoints):

            if i % 1000 ==0: 
                print('latitude = {}, r = {}rj'.format(coordinates[1], coordinates[0]/Rj))
            B_lh = self.field.CAN_sheet(coordinates[0], coordinates[1], coordinates[2])
            B_rh = self.helpful.B_S3LH_to_S3RH(B_lh[0], B_lh[1], B_lh[2])
            s3rh_cords = self.helpful.S3LH_to_S3RH(coordinates[0], coordinates[1], coordinates[2])
            B_x, B_y, B_z = self.helpful.Bsph_to_Bcart(B_rh[0], B_rh[1] ,B_rh[2], s3rh_cords[0], s3rh_cords[1],s3rh_cords[2])
            B = np.array([B_x, B_y, B_z])
            
            Bunit = self.helpful.unit_vector_cart(B)
            dr = s3rh_cords[0] * 0.0001 #(*Rj) #THIS IS HOW WE UPPDATE THE COORDINATES - IF IT TAKES TOO LONG, THIS NEEDS CHANGING IF IT TAKES TOO LONG OR IS GETTING WEIRD CLOSE TO PLANET
            coordinates = self.helpful.sph_to_cart(s3rh_cords[0], s3rh_cords[1], s3rh_cords[2])
            change = dr * Bunit
            coordinates = np.add(coordinates, change)
            pointslower.append(coordinates)
            pr, ptheta, pphi = self.helpful.cart_to_sph(coordinates[0], coordinates[1], coordinates[2])
            rLH, thetaLH, phiLH = self.helpful.S3RH_to_S3LH(pr, ptheta, pphi)
            coordinates = [rLH,thetaLH,phiLH]

        print('\n upper hemipshere')
        coordinates = [30* Rj, 0, 0]
        for j in range(self.numPoints):
            if j % 1000 == 0: 
                print('latitude = {}, r = {}rj'.format(coordinates[1], coordinates[0]/Rj))
            B_lh = self.field.CAN_sheet(coordinates[0], coordinates[1], coordinates[2])
            B_rh = self.helpful.B_S3LH_to_S3RH(B_lh[0], B_lh[1], B_lh[2])#, hemipshere='upper')
            s3rh_cords = self.helpful.S3LH_to_S3RH(coordinates[0], coordinates[1], coordinates[2])
            B_x, B_y, B_z = self.helpful.Bsph_to_Bcart(B_rh[0], B_rh[1] ,B_rh[2], s3rh_cords[0], s3rh_cords[1],s3rh_cords[2])
            B = np.array([B_x, B_y, B_z])
            
            Bunit = self.helpful.unit_vector_cart(B)
            dr = s3rh_cords[0] * 0.0001 #(*Rj) #THIS IS HOW WE UPPDATE THE COORDINATES - IF IT TAKES TOO LONG, THIS NEEDS CHANGING IF IT TAKES TOO LONG OR IS GETTING WEIRD CLOSE TO PLANET
            coordinates = self.helpful.sph_to_cart(s3rh_cords[0], s3rh_cords[1], s3rh_cords[2])
            change = dr * - Bunit
            coordinates = np.add(coordinates, change)
            pointsupper.append(coordinates)
            pr, ptheta, pphi = self.helpful.cart_to_sph(coordinates[0], coordinates[1], coordinates[2])
            rLH, thetaLH, phiLH = self.helpful.S3RH_to_S3LH(pr, ptheta, pphi)
            coordinates = [rLH,thetaLH,phiLH]
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        upper = np.array(pointsupper)
        lower = np.array(pointslower)
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
        plt.title('Magnetic Field Trace of current sheet')
        #plt.legend()
        plt.savefig('images/mag_field_trace__current_sheet.png')
        plt.show()

test = current_trace(40000)
test.trace()