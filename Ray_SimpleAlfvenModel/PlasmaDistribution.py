#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:25:39 2018

@author: liciaray
"""

import numpy as np
import scipy.constants as const

class PlasmaDistribution():
    def __init__(self, magneticfieldconfig, planet, nEquatorial, ionTemp, ionMass):
        self._magneticfieldconfig = magneticfieldconfig
        self._planet = planet
        self._nEquatorial = nEquatorial
        self._ionTemp = ionTemp
        self._ionMass = ionMass
        self._density = np.zeros_like(self._magneticfieldconfig.zDipole)
        self._exponentFactor = np.zeros_like(self._magneticfieldconfig.zDipole)
        
    @property
    def scaleHeight_const(self):
        #H_0 from Bagenal & Delamere [2011]
        return np.sqrt(2.*const.e/(3.*const.proton_mass))/self._planet.OP
    
    @property
    def scaleHeight(self):
        #calculate scale height
        return self.scaleHeight_const*np.sqrt(np.divide(self._ionTemp,self._ionMass))/self._planet.RP
    
    @property 
    def density(self):
        return self._density
    
    @property
    def ionMass(self):
        return self._ionMass
    
    @property
    def exponentFactor(self):
        return self._exponentFactor
    
    def densityProfile(self):
        #calculate density profile
        for i in range(0,self.scaleHeight.shape[0]):
            self._exponentFactor[:,i] = np.square(self._magneticfieldconfig.sFromEquator[:,i]/self.scaleHeight[i])
            self._density[:,i] = self._nEquatorial[i]*np.exp(-self._exponentFactor[:,i])
            self._density[self._density<100] = 100
            