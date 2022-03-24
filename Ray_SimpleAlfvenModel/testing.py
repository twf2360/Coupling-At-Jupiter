#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
"""
Created on Fri Nov 24 22:23:37 2017

This program calls the Hill solver for a dipole magnetic field with 
a user-specified mass-loading rate. The radial extent of the solver 
can be adjusted as needed. 

@author: liciaray
"""

#import necessary libraries and functions
import matplotlib.pyplot as plt
from Planet import Planet
import BFlux as Bf  
from Grid import Grid
from Field import Field

# Select planet. Options are
#jupiter, saturn, or earth
planet = Planet.jupiter()

# set spatial resolution and bounds
# Generate the radial grid
dr = 1e4
r_inner = 5.0
r_outer = 100.0
g = Grid(dr,r_inner,r_outer,planet.RP)


# Initialise the planetary magnetosphere
# options are dipole (Bf.fluxDIP, Bf.bmDIP)
# or stretched field configuration from 
# Nichols & Cowley[2004] (Bf.fluxNC, Bf.bmCANKK)
planet.initMagnetosphere(g,Bf.fluxDIP, Bf.bmDIP)

#specify parameters / inputs for calculation
# radial mass transport rate (kg/s)
mdot = 2000.0
# density of current carrying electrons
dens_elec = 1e8
# initial energy of current carrying electrons
temp_elec = 2.5e3

# initialise the electric fields and hight-integrated
# current density arrays
field = Field(mdot,dens_elec, temp_elec, planet, g)

# calculate the plasma angular velocity profile &
# associated currents
morePhysics = field.calculateHill()

#if morePhysics:
#    print("J thermal exceeded")
#    field.transition()
    

# Plot the flux thing in a classy way
#plt.figure(1)
f, axarr = plt.subplots(4, sharex=True)
axarr[0].semilogy(g.r/planet.RP,planet.flux)
axarr[0].set_xlim(r_inner,r_outer)
axarr[0].set_ylabel('Flux Function')
#plt.subplot(412)
axarr[1].plot(g.r/planet.RP,1+field.omega/planet.OP)
axarr[1].set_xlim(r_inner,r_outer)
axarr[1].set_ylabel('Angular\nVelocity\n($\Omega_{p})^{-1}$')
#plt.subplot(413)
axarr[2].plot(g.r/planet.RP,field.elec_m)
axarr[2].set_xlim(r_inner,r_outer)
axarr[2].set_ylabel('Electric field\n(mV/m$^{2}$)')
#plt.subplot(414)
axarr[3].plot(g.r/planet.RP, field.jpar_i*1e9)
#axarr[3].set_ylim([-2e-9,np.amax(field.jpar_i)])
axarr[3].set_ylim([min(field.jpar_i)*1e9,max(field.jpar_i)*1e9])
axarr[3].set_xlim(r_inner,r_outer)
axarr[3].set_ylabel('Current\nDensity\n(nA m$^{-2}$)')
axarr[3].set_xlabel('Radius (R$_{J}$)')
plt.show()
ang_vel = 1+field.omega/planet.OP
print(ang_vel)
ang_vel = np.array(ang_vel)
radii = np.array(g.r/planet.RP)
print(radii)
np.save('D:/Tom/Documents/Uni/Physics/Masters/Coupling-At-Jupiter/angular_velocity_data/mdot_{}'.format(mdot),ang_vel, allow_pickle=True)
np.save('D:/Tom/Documents/Uni/Physics/Masters/Coupling-At-Jupiter/angular_velocity_data/radii{}'.format(mdot),radii, allow_pickle=True)
