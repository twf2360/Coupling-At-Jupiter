#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 16:30:16 2018

@author: liciaray
"""

import numpy as np

class MagneticFieldConfig():
    def __init__(self, lshell_resolution, planet, latitude_resolution):
        self._lshell_resolution = lshell_resolution
        self._planet = planet
        self._latitude_resolution = latitude_resolution
#        self._sDipole = np.zeros(shape=(int(self._lshell_resolution),self._latitude_resolution))
#        self._sFromEquator = np.zeros(shape=(len(self._lshell),self._resolution))
        self.generate_grid()
        self.calc_sDipole()
        self.surfaceCorrection()
     
        
    @property
    def colatitude(self):
        return self._colatitude
#        return np.arange(0,self._resolution)/self._resolution*np.pi

    @property
    def lshell(self):
        return self._lshell
    
    def generate_grid(self):
        self._lshell, self._colatitude = np.meshgrid(np.linspace(2,102,self._lshell_resolution),np.linspace(0,180,self._latitude_resolution)) 
    
    @property
    def rDipole(self):
        return np.multiply(self.lshell,np.square(np.sin(self.colatitude*np.pi/180.)))
#        return np.outer(self._lshell,np.square(np.sin(self.colatitude)))
    
    @property
    def bMagnitude(self):
        return self._planet.BP/self._lshell*np.sqrt(1.0+3.0*np.square(np.cos(self.colatitude*np.pi/180)))
    
    @property
    def xDipole(self):
        return np.multiply(self.rDipole,np.sin(self.colatitude*np.pi/180))
    
    @property
    def zDipole(self):
        return np.multiply(self.rDipole,np.cos(self.colatitude*np.pi/180))
    
    @property
    def sDipole(self):
        return self._sDipole
    
    @property
    def dS(self):
        return self._dS
    
    @property
    def sFromEquator(self):
        return self._sFromEquator
#    
    def calc_sDipole(self):
        dx = np.diff(self.xDipole,axis=0)
        dz = np.diff(self.zDipole,axis=0)
        self._dS = np.sqrt(np.square(dx)+np.square(dz))
        self._dS = np.insert(self._dS,0,0,axis=0)
        
        self._sFromEquator = np.zeros_like(self._lshell)

        #calculate the length of the field line and distance from the equatorial plane  
        self._sDipole=np.cumsum(self._dS, axis=0)
        for i in range(0, self._lshell.shape[1]):           
            self._sFromEquator[:,i] = np.abs(self._sDipole[:,i]-self._sDipole[int(self._lshell.shape[0]/2),i])    
        
    
    def surfaceCorrection(self):
        self.rDipole[self.rDipole<1] = np.nan
        self.bMagnitude[self.rDipole<1] = np.nan
        self.xDipole[self.rDipole<1] = np.nan
        self.zDipole[self.rDipole<1] = np.nan