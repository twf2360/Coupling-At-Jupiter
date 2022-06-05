#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class to describe planets and provide basic properties. Most are constants.
Created on Fri Nov 24 22:39:56 2017

@author: liciaray
"""
import numpy as np

class Planet(object):
    def __init__(self,RP,MP,OP,BP,sp0):
        self._RP = RP
        self._MP = MP
        self._OP = OP
        self._BP = BP
        self._sp0 = sp0
        self._rad = None
        self._flux = None
        
    def initMagnetosphere (self, grid, fluxFcn,bmFcn):
        self._grid = grid
        self._flux = fluxFcn(self, self._grid.r) 
        self._flux_edge = fluxFcn(self,self._grid.r_edge)
        self._bm = bmFcn(self,self._grid.r)
        self._bm_edge = bmFcn(self, self._grid.r_edge)
        
  
    #Properties and checks    
    @property
    def sp0 (self):
        return self._sp0
    
    @sp0.setter
    def sp0 (self, sp0):
        if sp0 < 0:
            raise ValueError ('Conductance must be positive')
        else:
            self._sp0 = sp0
            
    @property
    def RP (self):
        return self._RP
  
    @property
    def MP (self):
        return self._MP
    
    @property
    def OP (self):
        return self._OP
    
    @property
    def BP (self):
        return self._BP
    
    @property
    def flux (self):
        return self._flux     
    
    @property
    def bm(self):
        return self._bm
    
    @property
    def flux_edge (self):
        return self._flux_edge     
    
    @property
    def bm_edge(self):
        return self._bm_edge

        
#assign radius, mass, angular frequency
#planetary magnetic field strength, pedersen conductance, 
#and mapping function
       
    #Jupiter 
    @classmethod
    def jupiter(cls):
        return cls(7.1492e7,1.89813e27,1.7735e-4,4.264e-4,0.1)
    
    #Saturn
    @classmethod
    def saturn(cls):
        return cls(6.0268e7,5.6834e26, 1.1157e-4,2.1136e-5,5.0)

    @classmethod
    def earth(cls):
        return cls(6.378137e6,5.9723e24,7.2722e-5,2.06e-5,1.0)
    
    #s distance
    @property
    def s(self):
        return np.sqrt(self._flux/self.BP)
    
    @property
    def theta(self):
        return np.arcsin(self.s/self.RP)            

    @property
    def theta_edge(self):
        return np.arcsin(np.sqrt(self._flux_edge/(self.BP))/self.RP)
    
    @property
    def bi(self):
        return np.multiply(self.BP,np.sqrt(1.0 + 3.0*np.square(np.cos(self.theta))))

    #sp distance
    @property
    def sp(self):
        return [self.sp0]*len(self._grid.r)
    
    @property
    def map_func(self):
        return 2.0*np.multiply(np.divide(self.BP,self.bm),np.divide(self.s,self._grid.r))
    
    @property
    def mirror_ratio(self):
        return np.divide(self.bi,self.bm)
    
    @property
    def mirror_ratio_edge(self):
        return np.divide(np.multiply(self.BP,np.sqrt(1.0 + 3.0*np.square(np.cos(self.theta_edge)))),self.bm_edge)
    
    @property
    def xSurface(self):
        return np.sqrt(1.- np.square(np.cos(np.arange(90)*np.pi/180.)))
    
    @property
    def zSurface(self):
        return np.sqrt(1.- np.square(np.sin(np.arange(90)*np.pi/180.)))
    