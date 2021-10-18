'''
If there is a spinning body, a coupling constant to something further away, how do the two interact? 
'''

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

''' 
starting with a 2d picture, looking down from above
'''

'''
some values of jupiter for ease of use 
'''
Mj = 1.898e27 #Kg
Rj = 7.14e7 #m
Tj = 32456 #s


class orbiter:
    def __init__(self, OrbitalRadius, SpinnerPeriod, coupling = 1, name='orbiter', SpinnerPosition = [0,0], theta = 0):
        self.OrbitalRadius = OrbitalRadius
        self.name = name
        self.coupling = coupling
        self.theta = theta
        CentralBody = self.spinner(Tj)
        self.position = np.array([self.OrbitalRadius * math.cos(self.theta), self.OrbitalRadius * math.sin(self.theta)], dtype = float)
        self.AngVel = CentralBody.AngVel * coupling

    def UpdatePosition(timestep):
        self.theta += self.AngVel * timestep #this will either break at 2pi or the sin/cos functions will deal with it nicely? 
        ''' 
        if self.theta > 2*math.pi 
        self.theta -= 2*math.pi
        '''
        self.position = np.array([self.OrbitalRadius * math.cos(self.theta), self.OrbitalRadius * math.sin(self.theta)], dtype = float)
        


    
    class spinner:
            '''
            the spinning body
            '''
            def __init__(self, period, name='spinner', position = [0,0]): 
                ''' 
                mass, radius, and period of rotation for the spinning body
                '''
                AngVel = 2*math.pi / period
                self.AngVel = AngVel
                self.position = np.array(position, dtype=float)
                self.name = name









    




