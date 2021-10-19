'''
If there is a spinning body, a coupling constant to something further away, how do the two interact? 
'''

import sys
import math
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

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

    def UpdatePosition(self, timestep):
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

    def calc(self, iterations, timestep):
        # add a delete previously existing file
        positions = []
        for i in range(iterations):
            positions.append(self.position)
            self.UpdatePosition(timestep)
        positions = np.array(positions) #need a way to save the names, and thge location of the central body too :D 

        np.save("simpleRotationsOutput.npy", positions, allow_pickle=True)

class plot:
    def __init__(self, plottype = 'animate'):
        if not os.path.exists("simpleRotationsOutput.npy"):
            print('no file to plot')
            sys.exit()
        points = np.load("simpleRotationsOutput.npy")
        Transposed_points = np.transpose(points)
        if plottype == 'animate':
            
            fig = plt.figure() 
            ax = plt.axes(xlim=(-1e9, 1e9), ylim=(-1e9, 1e9)) 
            ax.spines['left'].set_position('center')
            ax.spines['bottom'].set_position('center')
            # Eliminate upper and right axes
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            #ax.add_patch(Circle((0,0), Rj, color='y', zorder=100, label = "Jupiter")) <- add the central body in
            line, = ax.plot([], [], 'o-')

            def animate_init():
                line.set_data([], [])
                return line,
            def animate(i):
                x = Transposed_points[0][i]
                y = Transposed_points[1][i]
                line.set_data(x,y)
                #print(Transposed_points[0], Transposed_points[0][i])
                return line,

            anim = FuncAnimation(fig, animate, init_func=animate_init,frames=200, interval=20, blit=True)
            plt.show()

'''
test = orbiter(422e6, Tj)
test.calc(5000, 100)
'''

plottest = plot()


    











    




