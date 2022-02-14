'''
There seems to be functions that I keep repeating, and instead of writing the same thing out multiple times it makes more sense just to have them all here.

This will hopefully be added to as we go! 
'''
import numpy as np 
import math
import sys
from mag_field_models import field_models
Rj = 7.14e7
class HelpfulFunctions():

    def __init__(self, model = 'VIP4'):
        self.field = field_models()
        self.model = model
         

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


    def complex_mag_equator(self, r, phi_lh):
        ''' r in rj, theta colatitude, phi lh in rad''' 
        phi =  2 *np.pi - phi_lh #change phi to RH. 
        #phi = phi_lh
        guesses_degress = np.array([30,40,50,60,70,80,90,100,110,120,130, 140, 150], dtype = float)
        guesses_radians_1 = guesses_degress * np.pi/180
        oneDegreeInRadians = 1*np.pi/180
        def find_swap(angles):
            #print(angles)
            b_r_list = []
            #print(b_r_list)
            for i in range(len(angles)):
                B_r, B_theta, B_phi = self.field.Internal_Field(r, angles[i], phi , model=self.model) #calculates the magnetic field due to the internal field in spherical polar that point)
                B_current = self.field.CAN_sheet(r, angles[i], phi) #calculates the magnetic field due to the current sheet in spherical polar
                B_notcurrent = np.array([B_r, B_theta, B_phi]) 
                B_overall = np.add(B_current, B_notcurrent)
                #print(B_overall[0])
                b_r_list.append(B_overall[0])
                #print(B_overall)
                #print(b_r_list)
            crossing = np.where(np.diff(np.sign(b_r_list)))[0]
            b_r_list = []
            #print(crossing)
            #print(b_r_list)       
            return angles[crossing], angles[crossing+1]
        stop1, stop2 = find_swap(guesses_radians_1)

        #print(stop1*180/np.pi, stop2*180/np.pi)
        guesses_2 = np.arange(stop1, stop2 + oneDegreeInRadians, oneDegreeInRadians)
        #print(guesses_2)

        stop3, stop4 = find_swap(guesses_2)
        #print(stop3*180/np.pi, stop4*180/np.pi)
        guesses_3 = np.arange(stop3, stop4+ 0.1* oneDegreeInRadians, 0.1*oneDegreeInRadians )
        answer_low, answer_high = find_swap(guesses_3)
        answer = (answer_high + answer_low) /2
        #print(answer_low*180/np.pi, answer_high*180/np.pi)
        #print(answer[0]*180/np.pi)
        return answer[0]
    

    def calc_furthest_r(self, points):
        ''' calc L shell ''' 
        rs = []
        for point in points:
            x = point[0]
            y = point[1]
            z = point[2]
            rs.append(np.sqrt(x**2 + y**2 + z**2))
        largest_r = max(rs)
        largest_r_rj = largest_r/Rj
        return largest_r_rj
       

  
    
    
    
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


        
'''
test = HelpfulFunctions()
print(test.complex_mag_equator(10,111*np.pi/180) * 180/np.pi)
'''