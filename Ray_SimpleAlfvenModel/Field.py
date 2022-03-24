#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:13:26 2017

@author: liciaray
"""

import numpy as np
import scipy.constants as const

class Field():
    def __init__(self, mdot, dens_elec, temp_elec, planet, grid):
        self._mdot = mdot
        self._dens_elec = dens_elec
        self._temp_elec = temp_elec
        self._planet = planet
        self._grid = grid
        self._gridSize = len(self._grid.r)
        self._omega = np.zeros(len(self._grid.r))
        self._elec_m = np.zeros(len(self._grid.r))
        self._elec_i = np.zeros(len(self._grid.r))
        self._k_m = np.zeros(len(self._grid.r))
        self._k_i = np.zeros(len(self._grid.r))
        self._jpar_m = np.zeros(len(self._grid.r_edge))
        self._jpar_i = np.zeros(len(self._grid.r_edge))

        
    @property
    def omega (self):    
        return self._omega
        
    @property
    def elec_m (self):   
        return self._elec_m
    
    @property
    def elec_i (self):
       return self._elec_i
    
    @property
    def k_m (self):
        return self._k_m
    
    @property
    def k_i (self):
        return self._k_i
    
    @property
    def jpar_m (self):
        return self._jpar_m
    
    @property
    def jpar_i (self):
        return self._jpar_i
    
    @property
    def j_par_th(self):
        return self._dens_elec*const.e*np.sqrt(self._temp_elec*const.e/(2.0*np.pi*const.m_e))
    
        
        
    #initialize a corotating inner boundary
    def calculateHill(self):
        
        i = 0

#        jpar = 0 
        
        #define factors for hill calculation
        aa = (-2.0*(np.multiply(np.multiply(self._planet.sp,self._planet.map_func),
                                np.multiply(self._planet.bm,self._planet.s))))
        omega_div_r = 2.0*np.divide(self._planet.OP,self._grid.r)
        bm_div_mdot = 2.0*np.pi*np.divide(self._planet.bm,self._mdot)
        two_div_r = np.divide(2.0,self._grid.r)
        rm_div_2rdr = np.multiply(self._planet.mirror_ratio_edge,np.divide(0.5,np.multiply(self._grid.r_edge[i],self._grid.dr)))
        
 #       while jpar < self.j_par_th and i < (self._gridSize-1):
        while i < (self._gridSize-1):
 #           if i%1000 == 0: 
 #               print("i = ", i)
            #find domega/dr
            domdr = (np.multiply(bm_div_mdot[i],self._k_m[i]) - 
                     omega_div_r[i] - np.multiply(self._omega[i],two_div_r[i]))
         
            #predict angular velocity and current at next spatial step
            omp1 = self._omega[i] + self._grid.dr*domdr
            kmp1 = (aa[i+1]*omp1)
       
            
            #predict domega/dr at advanced step
            domdrp1 = (np.multiply(bm_div_mdot[i+1],kmp1) - 
                       omega_div_r[i+1] - np.multiply(omp1,two_div_r[i+1]))
        
            #set omega for next spatial step 
            self._omega[i+1] = self._omega[i] + self._grid.dr*0.5*(domdr + domdrp1)
            self._k_m[i+1] = (aa[i+1]*self._omega[i+1])
            
            #if i > 1000 then calculate the field-aligned
            #currents using div.J = 0
#            if i>5000 and i%500==0:
#                jpar = np.multiply(rm_div_2rdr[i],(np.multiply(self._k_m[i+1],self._grid.r[i+1]) - 
#                                     np.multiply(self._k_m[i],self._grid.r[i])))
            
            #iterate i
            i+=1
            
        self._elec_m = np.multiply(np.multiply(self._omega,self._grid.r),self._planet.bm)
        self._elec_i = np.multiply(self._planet.map_func,self._elec_m)
        self._k_i = np.multiply(self._planet.sp,self._elec_i)
        self._jpar_i = np.multiply(rm_div_2rdr,np.ediff1d(np.multiply(self.k_m,self._grid.r), to_end=0))
        self._jpar_m = np.divide(self._jpar_i,self._planet.mirror_ratio_edge)
        
        if i < (self._gridSize-1):
            self._ii = i
            return True
        else:
            return False
        
        
     
      