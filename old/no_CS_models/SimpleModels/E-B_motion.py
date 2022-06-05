'''
how can we model the motion of a charged particle? 

'''
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

Me = 9.11e-31
Qe = -1.6e-19

class particle:
    def __init__(self, mass,charge, position = [0,0,0], StartVel = [0,0,0]):
        self.mass = mass
        self.charge = charge
        self.position = position
        self.velocity = StartVel

    def cromer_update(self, dv, dt):
        '''
        updates the position of the ball after a time dt, using the euler-cromer approximation
        dt = the timestep between iterations used by the calculator 
        dv = the change in velocity that has been calculated using the calculator
        '''
        self.velocity += dv
        self.position += self.velocity * dt



class Bfield:
    '''
    define the magnetic field 
    for now, constant magnitude constrained to x, y, z 
    '''
    def __init__(self, magnitude, direction = 'z'):
        self.magnitude = magnitude
        if direction == 'x':
            self.B = [magnitude,0,0]
        
        if direction == 'y':
            self.B = [0,magnitude,0]

        if direction == 'z':
            self.B = [0,0,magnitude]

class Efield:
    '''
    define the magnetic field 
    for now, constant magnitude constrained to x, y, z 
    '''
    def __init__(self, magnitude, direction = 'z'):
        self.magnitude = magnitude
        if direction == 'x':
            self.E = [magnitude,0,0]

        if direction == 'y':
            self.E = [0,magnitude,0]

        if direction == 'z':
            self.E = [0,0,magnitude]
        


class motion:
    def __init__(self, timestep, iterations):
        self.Bfield = Bfield(1e-9)
        self.Efield = Efield(1e-9)
        self.particle = particle(Me, Qe)
        self.iterations = iterations
        self.timestep = timestep

    def calculate_acceleration(self):
        q = self.particle.charge
        e = self.Efield.E
        v = self.particle.velocity
        b = self.Bfield.B
        m = self.particle.mass

        F = q*(e + np.cross(v, b))
        a = F/m 
        dv = a * self.timestep
        return dv

    def animate(self):
        
        def update():
            dv = self.calculate_acceleration()
            self.particle.cromer_update(dv, self.timestep)
            positions.append(self.particle.position)
        
        positions_transpose = np.transpose(positions)
        plt.style.use('dark_background')
        fig = plt.figure() 
        ax = plt.axes() 
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        time_template = 'time ={}s'
        time_text = ax.text(0.05,0.9, '', transform=ax.transAxes)
        particles, = ax.plot([], [])
        
        def animate_init():
            x = self.
            particles.set_data([], [])
            
            return 
        def animate(i):
            
            
            time_text.set_text(time_template.format(i*self.timestep)) 
           
            return  time_text

        anim = FuncAnimation(fig, animate, init_func=animate_init,frames=200, interval=20, blit=True)
        plt.show()
        anim.save('images/SimpleEBMotion.gif')

    

    

test = motion(0.00001, 1000)
test.plot()
        
        
    
    

    