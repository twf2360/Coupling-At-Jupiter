'''
Just defining a particle, to then define it's motion in E fields, B fields, and then both? 
'''

class particle:
    
    def __init__(self, mass, charge, r, theta, phi, velocity, acceleration):
        '''
        defining the particle, position is defined using spherical co-ordinates - can always change to xy at a later date.
        '''

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
    '''
class field:
    '''
    define the field that the particle is moving through
    '''

    ''' need to think about how we're doing this - field lines? defining start and end? '''
        


class plot: 
    '''
    plot the particle motion through a field 
    '''