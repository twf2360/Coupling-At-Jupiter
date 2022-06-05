# -*- coding: utf-8 -*-
"""

This is a toy model to calculate the Alfven travel
time along a magnetic field line. This is an important
parameter to know in MI coupling as the Alfven travel time 
is the effective communication time between the atmosphere 
and the magnetodisc.

@author: liciaray
"""

#import necessary libraries and functions
import matplotlib.pyplot as plt
from Planet import Planet
from MagneticFieldConfig import MagneticFieldConfig
from PlasmaDistribution import PlasmaDistribution
from PlanetarySystem import PlanetarySystem
import numpy as np


#define planet
planet = Planet.jupiter()

##set up magnetosphere
#select resolution
lshell_resolution = int(20)
latitude_resolution = int(720)
dipole = MagneticFieldConfig(lshell_resolution, planet, latitude_resolution)

##################################################

#set equatoral plasma mass, temperature, and density.

n0 = 20. #per cc
n0 = n0*1e6 #per m^3
ionTemp_0 = 100 #in eV
ionMass_0 = 20 #in amu


#play around with the function forms with distance from the planet
ionMass = ionMass_0/np.sqrt(dipole.lshell[0,:]) 
nEquatorial = n0*(dipole.lshell[0,:]**-1.5)
ionTemp = ionTemp_0*np.sqrt(dipole.lshell[0,:])


#workhorse of density and alfven speed solvers
plasma = PlasmaDistribution(dipole,planet,nEquatorial,ionTemp,ionMass)
plasma.densityProfile()
vAlfven = PlanetarySystem(planet,dipole,plasma)


###################################################
#Plot parameters of interest

#Magnetic field
fig,ax = plt.subplots(1)
plt.plot(dipole.xDipole,dipole.zDipole)
ax.add_artist(plt.Circle((0,0),1,color='orange', zorder=100))
C1 = plt.contourf(dipole.xDipole,dipole.zDipole,dipole.bMagnitude,20, alpha = 0.4, antialiased=True)
ax.set_xlabel('x (R$_{P}$)')
ax.set_ylabel('z (R$_{P}$)')
ax.set_title('Dipole Field Lines and B Strength')
cbar = plt.colorbar(C1)
cbar.ax.set_ylabel('|B| (Tesla)')

#Distance along field
fig,ax = plt.subplots(1)
plt.plot(dipole.colatitude, dipole.sFromEquator)
ax.set_xlabel('colatitude')
ax.set_ylabel('z (R$_{P}$)')
ax.set_title('Distance Along the Field from the Equator')

#Plasma density
fix,ax = plt.subplots(1)
plt.plot(dipole.xDipole,dipole.zDipole,color='w')
ax.add_artist(plt.Circle((0,0),1,color='orange', zorder=100))
C2 = plt.contourf(dipole.xDipole,dipole.zDipole,np.log10(plasma.density),80, alpha = 0.7, antialiased=True)
ax.set_title('log$_{10}$ Plasma Density')
ax.set_xlabel('x (R$_{P}$)')
ax.set_ylabel('z (R$_{P}$)')
cbar = plt.colorbar(C2)
cbar.ax.set_ylabel('n (m$^{-3}$)')

#Alfven speed
fix,ax = plt.subplots(1)
C3 = plt.contourf(dipole.xDipole,dipole.zDipole,np.log10(vAlfven.nonRelativisticSpeed),80)
ax.add_artist(plt.Circle((0,0),1,color='orange', zorder=100))
ax.set_title('log$_{10}$ Alfven Speed')
ax.set_xlabel('x (R$_{P}$)')
ax.set_ylabel('z (R$_{P}$)')
cbar = plt.colorbar(C3)
cbar.ax.set_ylabel('log(m s$^{-1}$)')

#Alfven transit time contribution
fix,ax = plt.subplots(1)
C4 = plt.contourf(dipole.xDipole,dipole.zDipole,vAlfven.transitContribution,80)
ax.add_artist(plt.Circle((0,0),1,color='orange', zorder=100))
ax.set_title('Contribution to Alfven transit time')
ax.set_xlabel('x (R$_{P}$)')
ax.set_ylabel('z (R$_{P}$)')
cbar = plt.colorbar(C4)
cbar.ax.set_ylabel('time (s)')

#Alfven transit time from equaotor
fix,ax = plt.subplots(1)
C5 = plt.contourf(dipole.xDipole,dipole.zDipole,vAlfven.tFromEquator,100)
ax.add_artist(plt.Circle((0,0),1,color='orange', zorder=100))
ax.set_title('Alfven transit time from Equator')
ax.set_xlabel('x (R$_{P}$)')
ax.set_ylabel('z (R$_{P}$)')
cbar = plt.colorbar(C5)
cbar.ax.set_ylabel('time (s)')

#Alfven transit time in total
fix,ax = plt.subplots(1)
plt.plot(dipole.lshell,vAlfven.transitTime)
plt.plot(dipole.lshell[0,:],vAlfven.transitTime[latitude_resolution-1,:])
ax.set_title('Alfven travel time vs L-Shell')
ax.set_xlabel('L-Shell')
ax.set_ylabel('Alfven Travel Time,pole to pole (s)')
