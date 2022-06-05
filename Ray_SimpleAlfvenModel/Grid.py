#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 22:37:45 2017

@author: liciaray
"""
import numpy as np


class Grid:
    def __init__ (self, dr, r_inner, r_outer, rPlanet):
        self._dr = dr
        self._r_inner = r_inner
        self._r_outer = r_outer
        self._rPlanet = rPlanet
        
    @property
    def dr(self):
        return self._dr
    
    @property
    def r_inner(self):
        return self._r_inner
    
    @property
    def r_outer(self):
        return self._r_outer
    
    @property
    def n_steps(self):
        return (self._r_outer - self._r_inner)*self._rPlanet/self._dr
    
    @property
    def r(self):
        return np.arange(self.n_steps)*self._dr+self._r_inner*self._rPlanet
       
    #define edge as leading edge of cell
    #note that this is different to IDL model    
    @property
    def r_edge(self):
        return np.arange(self.n_steps)*self._dr + self._r_inner*self._rPlanet + self._dr*0.5