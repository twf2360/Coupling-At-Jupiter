#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 21:47:16 2017

Create the Jupiter flux function

@author: liciaray
"""

from scipy.special import gammaincc, gamma
import numpy as np

#Jupiter flux function
def fluxNC(self, rad):
    #returns the equatorial flux function defined in nichols and cowley 2004
    #function has units of nT * RJ^2 so result is converted to SI (T * m^2) here
    b0     = 3.335e5             #nT
    roe0   = 14.501*self.RP
    a      = 5.4e4               #nT
    m      = 2.71
    r_jup3 = np.power(self.RP,3)   #m^3
    r_jup2 = np.power(self.RP,2) #m^2
    f_inf  = 2.841e4             #flux at infinity in nT*Rj^2
    aa = -2./5.    
    rexp = 5./2.

    #uses gamma function identities to calculate upper gamma function
    upper_gamma = (1./aa)*(gamma(aa+1.)*gammaincc(aa+1,np.power((rad/roe0),rexp)) - 
                   np.power(np.power((rad/roe0),rexp),aa)*
                   np.exp(-(np.power((rad/roe0),rexp))))
    
    return (np.multiply(f_inf,r_jup2) + np.multiply(np.divide(np.multiply(b0,r_jup3),(2.5*roe0)), \
            upper_gamma) + a/(m-2.)*np.multiply(np.power((self.RP/rad),(m-2.)),r_jup2))*1e-9
    
    
def bmCANKK(self,rad):
    
    b0     = 3.335e5             #nT
    roe0   = 14.501*self.RP
    a      = 5.4e4               #nT
    m      = 2.71
        
    return (b0*np.power(np.divide(self.RP,rad),3)*np.exp(-(np.power(np.divide(rad,roe0),2.5))) 
            + a*np.power(np.divide(self.RP, rad),m))*1e-9
    
             
def bmDIP(self,rad):
    return self.BP*np.power(np.divide(self.RP,rad),3)

def fluxDIP(self,rad):
    return self.BP*np.divide(np.power(self.RP,3),rad)

