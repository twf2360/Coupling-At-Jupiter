'''   
Testing Chris' mag field models code as I'm getting a value for B that is much larger than expected, leading to a much larger VA than expected

I took out the pytest framework and I'm just doing it as prints instead for simplicity. 
'''
import numpy as np 
from mag_field_models import field_models
import math

''' 
First Testing Dipole Model
'''
field_models = field_models()
def test_dipole():
    B0 = 4.17 * (10 ** -4)
    r = 20 #in Rj
    colatitude = np.pi/2 
    righthand_phi = 0
    test_point = [r, colatitude, righthand_phi]
    '''
    calculating what it would be as a dipole manually, using magnetic field strength at equator as B0 = 417 micro Tesla
    ''' 
    dipole_result = B0 * ((1/r)**3)
    
    
    chris_result_vector = np.array(field_models.Internal_Field(r, colatitude, righthand_phi, model='dipole'))
    chris_result_magnitude = np.linalg.norm(chris_result_vector)
    ''' 
    i should've turned into into cartestian before doing this, but it got the point across 
    '''
    divided_result = chris_result_magnitude/dipole_result
    factor = math.floor(math.log10(divided_result))
    print(' \n Dipole Result: {}T \n Chris Result: {}T \n Magnitude difference: {}'.format(dipole_result, chris_result_magnitude, factor))
    
test_dipole()


''' 
it was at this point i realised that chris' code is in nanotesla.
'''