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

    def __init___(self, numPoints):
        self.field = field_models()
        self.numPoints = numPoints

        

    def trace(self):
        start = []
        for i in range(self.numPoints):

