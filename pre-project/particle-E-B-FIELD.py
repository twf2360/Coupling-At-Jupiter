'''
Just defining a particle, to then define it's motion in E fields, B fields, and then both? 
'''
import numpy as np
class particle:
    
    def __init__(self, mass, charge, r, theta, phi, velocity, acceleration):
        #theta = colatitude, phi = azimuthal
        '''
        defining the particle, position is defined using spherical co-ordinates - can always change to xy at a later date.
        '''
        self.mass = mass
        self.charge = charge 
        self.radial = r
        self.azimuthal = phi
        self.colatitude = theta
        self.velocity = velocity
        self.acceleration = acceleration

    def kinetic_energy(self):
        '''
        returns the kinetic energy of the particle at a given instance

        will need changing as the particle is not defined by xy
        '''
        ke = 0.5*self.mass*np.linalg.norm(self.velocity**2)
        return ke 


class field_line:
    '''
    initial precursor to the field class - how does a particle move on a single field line?
    it might be easier to model this using xy co-ordinates for the particle instead of using spherica; 
    '''
class dipole_field:
    '''
    define the field that the particle is moving through
    '''
    

        


class plot: 
    '''
    plot the particle motion through a field 
    '''