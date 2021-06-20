'''
Just defining a particle, to then define it's motion in E fields, B fields, and then both? 
'''

class particle:
    
    def __init__(self, mass, charge, r, theta, phi):
        '''
        defining the particle, position is defined using spherical co-ordinates - can always change to xy at a later date.
        '''