#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 22:35:41 2018

@author: liciaray
"""
import scipy.constants as const
import numpy as np

class PlanetarySystem():
    def __init__(self, planet, magnetosphere, plasmadist):
        self._planet = planet
        self._magnetosphere = magnetosphere
        self._plasmadist = plasmadist
        self.calculateTransitTime()

    @property
    def nonRelativisticSpeed(self):
        return np.divide(self._magnetosphere.bMagnitude,np.sqrt(const.mu_0*self._plasmadist.density*self._plasmadist.ionMass*const.proton_mass))
   
    @property
    def transitTime(self):
        return self._transitTime
    
    @property
    def tFromEquator(self):
        return self._tFromEquator
    
    @property
    def transitContribution(self):
        return self._transitContribution
    

    def calculateTransitTime(self):
        self._tFromEquator = np.zeros_like(self._magnetosphere.lshell)
        
        #calculate the contribution to the Alfven transit time at all locations
        self._transitContribution = np.divide(self._magnetosphere.dS*self._planet.RP,self.nonRelativisticSpeed)
        self._transitTime = np.cumsum(self._transitContribution,axis=0)

        #calculate the length of the field line and distance from the equatorial plane      
        for i in range(0, self._magnetosphere.lshell.shape[1]):            
            self._tFromEquator[:,i] = np.abs(self._transitTime[:,i]-self._transitTime[int(self._magnetosphere.lshell.shape[0]/2),i]) 
            
   