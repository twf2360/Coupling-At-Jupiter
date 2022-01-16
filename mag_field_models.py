# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 09:47:40 2019

@author: chris
"""

import numpy as np
import math
import scipy
import scipy.special


class field_models(object):
    
    # ================================================
    
    def cart_sph(self, x,y,z):
        r = np.sqrt(x**2+y**2+z**2)
        theta = np.arctan2(np.sqrt(x**2+y**2),z)
        phi = np.arctan2(y,x)
        return r, theta, phi
                 
         
    def cart_cyl(self, x,y,z):
        rho = np.sqrt(x**2+y**2)
        theta = np.arctan2(y,x)
        return rho,theta,z        
         
         
    def cyl_cart(self, rho, theta, z):
        x = rho*np.cos(theta)
        y = rho*np.sin(theta)
        return x,y,z
         
    def cyl_sph(self, rho, theta, z):
        r = np.sqrt(rho**2+z**2)
        thetasph = theta
        phi = np.arctan2(rho,z)
        return r, thetasph, phi
         
    def sph_cart(self, r, theta, phi):
        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)
        return x,y,z
         
    def sph_cyl(self, r, theta, phi):
        rho = r*np.sin(theta)
        thetacyl = phi
        z = r*np.cos(theta)        
        return rho, thetacyl, z
    
     
    # ================================================
    
    
    def CAN_sheet(self,r,theta,phi):
        '''
        DESCRIPTION:
            Connerney, Acuna & Ness Current sheet model. 
            IDL code origionaly from Fran Bagenal, recieved from Licia Ray and 
            translated into Python by Chris Lorch.
            Model uses the analytical aproximations given in the appendix of 
            Connerney, Acuna & Ness, [1981].
            
        INPUT:
            r - systemIII radius in Rj
            theta - systemIII colatitude in radians
            phi - systemIII west longitude in radians
            
        OUTPUT:
            B - array containing the B field components in a spherical system III 
                reference frame.
        
        '''
        
        #set current sheet parameters
        a = 5.0 # inner current sheet boundary
        b = 50.0 # outer current sheet boundary
        D = 2.5 # current sheet half thickness
        c = 225 # (mu_0*I_0)/2
        DtoR = np.pi/180
        theta_CS = 9.6*DtoR #Current sheet tilt
        CS_pm = 158*DtoR # Current sheet prime meridian
        
    
        # shortened for easier reading
        ct = np.cos(theta_CS)
        st = np.sin(theta_CS)
        cp = np.cos(CS_pm)
        sp = np.sin(CS_pm)
        
        #Transform from spherical sysIII to cylindrical Current sheet frame
        
        x,y,z = self.sph_cart(r,theta,phi)
        
        #rotate cartesian s3 into cartesian CS ref frame
        x_cs = cp*ct*x+sp*ct*y-st*z
        y_cs = -sp*x+cp*y
        z_cs = st*cp*x+st*sp*y+ct*z
        
        #transform from cartesian CS to cylindrical CS
        rho_cs,phi_cs,z_cs = self.cart_cyl(x_cs,y_cs,z_cs)
        phi = phi%(np.pi*2) # ensure phi is positive
        
        
        #determines which analytical approximations to make
        if rho_cs < a: # inside region 1
            F1 = np.sqrt((z_cs-D)**2+a**2)
            F2 = np.sqrt((z_cs+D)**2+a**2)
            F3 = np.sqrt(z_cs**2+a**2)
            Brho = (rho_cs/2)*(1/F1 - 1/F2)
            Bz = 2*D/F3-((rho_cs**2)/4)*((z_cs-D)/F1**3 - (z_cs+D)/F2**3)
        else:
            F1 = np.sqrt((z_cs-D)**2+rho_cs**2)
            F2 = np.sqrt((z_cs+D)**2+rho_cs**2)
            Brho = (F1-F2+2*D)/rho_cs #There was an error here.(F1+F2)=>(F1-F2)
            if abs(z_cs) > D and z_cs <0: # inside region 2
                Brho = (F1 - F2 - 2*D)/rho_cs
            elif abs(z_cs) < D: # inside region 3
                Brho = (F1-F2+2*z_cs)/rho_cs
            Brho = Brho - ((a**2*rho_cs)/4)*(1/F1**3-1/F2**3)
            Bz = 2*D/np.sqrt(z_cs**2+rho_cs**2) - ((a**2)/4)*((z_cs-D)/F1**3 - (z_cs+D)/F2**3)
            
        F1 = np.sqrt((z_cs-D)**2+b**2)
        F2 = np.sqrt((z_cs+D)**2+b**2)
        F3 = np.sqrt(z_cs**2+b**2)
        
        Brho2 = (rho_cs/2)*(1/F1 - 1/F2)
        Bz2 = 2*D/F3 - ((rho_cs**2)/4)*( (z_cs-D)/F1**3 - (z_cs+D)/F2**3 )
        
        
        # converts back into systemIII reference frame    
        Brho = Brho - Brho2
        Bz = Bz - Bz2   
        Bphi = np.sin(phi_cs)*Brho
        Brho = np.cos(phi_cs)*Brho
        
        #converts cylindrical vector to cartesian
        B1 = cp*ct*Brho - sp*Bphi + st*cp*Bz
        B2 = sp*ct*Brho + cp*Bphi + st*sp*Bz
        B3 =   -st*Brho + ct*Bz
        
        cth = np.cos(theta)
        sth = np.sin(theta)
        cph = np.cos(phi)
        sph = np.sin(phi)
        
        # converts cartesian to spherical
        B = [[],[],[]]
        B[0] = sth*cph*B1 + sth*sph*B2 + cth*B3
        B[1] = cth*cph*B1 + cth*sph*B2 - sth*B3
        B[2] = -sph*B1 + cph*B2
        
        B = np.array(B)*c
        
        return B
    
    # ================================================
    
    def SchmidtLegendre(self,n,m,theta): 
        import scipy
        #Binomial expansion of (cos(theta)^2-1)^n
        #determining coefficients of the binomial expansion (substitute in own theta)
        coeffs = []
        k = 0
        while k<=n:
            coeffs.append(scipy.special.comb(n,k)) #chaged by twf2360 due to depreciated scipy.misc.comb
            k += 1

        
        #differentiating method, changing the coefficients...
        
        order = n+m
        diffterm = coeffs
        for i in range(len(coeffs)):
            for j in range(order):
                diffterm[i] = diffterm[i]*(((2*n)-(2*i))-j)
        
                
        #determine the value of the equation
        terms = []
        for i in range(n+1):
            #determines the value of each term in the equation and sign
            term = ((-1)**(i))*coeffs[i]*np.cos(theta)**(((2*n)-(2*i))-order) #(-1)**(i) determines the sign
            terms.append(term)
            #sum of the terms gives the value of hte binomial expansion
            equation = sum(terms)
        
        
        
        #Schmidt quasi-normalisation coefficient
        SmtNorm = np.sqrt((2*math.factorial(n-m))/(math.factorial(n+m)))
        
        
        #Equation for the legendre polynomial
        if m ==0:
            Pn = (1/((2**n)*math.factorial(n)))*((np.sin(theta))**(m/2))*equation
        else:
            Pn = SmtNorm*(1/((2**n)*math.factorial(n)))*((np.sin(theta)**2)**(m/2))*equation
        return Pn
    
    
    # ================================================
    
    def Internal_Field(self, r, theta, phi, model = None):
        '''
        DESCRIPTION:
             Jupiter internal field model.
             
        INPUT:
            All inputs should be in system III right hand coordinates.
            
            r - range in Rj
            theta - Colatitude in radians
            phi - West longidude in radians
            model - Internal field model, choices include:
                JRM09
                VIP4
                VIT4
                Ulysses 17ev
                V1-17ev
                O6
                O4
                SHA
		        dipole
                
                
        OUTPUT: (in nT)
            Brint - the radial component of the internal field
            Btint - the theta component of the internal field
            Bpint - the azimuthal component of the internal field
            
        '''
        
        
        ##-------------------------------------------------------------------------
        ## Coefficients for each model
        if model == 'JRM09':
            
            order = 10
            g = { '1,0':410244.7,'1,1':-71498.3,
                 '2,0':11670.4,'2,1':-56835.8,'2,2':48689.5,
                 '3,0':4018.6,'3,1':-37791.1,'3,2':15926.3,'3,3':-2710.5,
                 '4,0':-34645.4,'4,1':-8247.6,'4,2':-2406.1,'4,3':-11083.8,
                 '4,4':-17837.2, '5,0':-18023.6,'5,1':4683.9,'5,2':16160.0,
                 '5,3':-16402.0,'5,4':-2600.7,'5,5':-3660.7,'6,0':-20819.6,
                 '6,1':9992.9,'6,2':11791.8,'6,3':-12574.7,'6,4':2669.7,
                 '6,5':1113.2,'6,6':7584.9,'7,0':598.4,'7,1':4665.9,'7,2':-6495.7,
                 '7,3':-2516.5,'7,4':-6448.5,'7,5':1855.3,'7,6':-2892.9,
                 '7,7':2968.0,'8,0':10059.2,'8,1':1934.4,'8,2':-6702.9,'8,3':153.7,
                 '8,4':-4124.2,'8,5':-867.2,'8,6':-3740.6,'8,7':-732.4,
                 '8,8':-2433.2,'9,0':9671.8,'9,1':-3046.2,'9,2':260.9,'9,3':2071.3,
                 '9,4':3329.6,'9,5':-2523.1,'9,6':1787.1,'9,7':-1148.2,'9,8':1276.5,
                 '9,9':-1976.8,'10,0':-2299.5,'10,1':2009.7,'10,2':2127.8,
                 '10,3':3498.3,'10,4':2967.6,'10,5':16.3,'10,6':1806.5,
                 '10,7':-46.5,'10,8':2897.8,'10,9':574.5,'10,10':1298.9
                 }
            
            h = {'1,0':0.,'1,1':21330.5,
                 '2,0':0.,'2,1':-42027.3,'2,2':19353.2,
                 '3,0':0.,'3,1':-32957.3,'3,2': 42084.5,'3,3':-27544.2,
                 '4,0':0.,'4,1':31994.5,'4,2':27811.2,'4,3':-926.1,'4,4':367.1,
                 '5,0':0.,'5,1':45347.9,'5,2':-749.0,'5,3':6268.5,'5,4':10859.6,
                 '5,5':9608.4,'6,0':0.,'6,1':14533.1,'6,2':-10592.9,'6,3':568.6,
                 '6,4':12871.7,'6,5':-4147.8,'6,6':3604.4,'7,0':0.,'7,1':-7626.3,
                 '7,2':-10948.4,'7,3':2633.3,'7,4':5394.2,'7,5':-6050.8,
                 '7,6':-1526.0,'7,7':-5684.2,'8,0':0.,'8,1':-2409.7,'8,2':-11614.6,
                 '8,3':9287.0,'8,4':-911.9,'8,5':2754.5,'8,6':-2446.1,'8,7':1207.3,
                 '8,8':-2887.3,'9,0':0.,'9,1':-8467.4,'9,2':-1383.8,'9,3':5697.7,
                 '9,4':-2056.3,'9,5':3081.5,'9,6':-721.2,'9,7':1352.5,'9,8':-210.1,
                 '9,9':1567.6,'10,0':0.,'10,1':-4692.6,'10,2':4445.8,'10,3':-2378.6,
                 '10,4':-2204.3,'10,5':164.1,'10,6':-1361.6,'10,7':-2031.5,
                 '10,8':1411.8,'10,9':-714.3,'10,10':1676.5
                 }
            
        elif model == 'VIPAL':
        
            order = 5
            g = {'1,0':420000.,'1,1':-69750,
                 '2,0':64410,'2,1':-86720,'2,2':95980,
                 '3,0':-10580,'3,1':-59000,'3,2': 63220,'3,3':46710,
                 '4,0':-74660.,'4,1':32820,'4,2':-33800,'4,3':18260,'4,4':-14290,
                 '5,0':-6600,'5,1':7370,'5,2':-17110,'5,3':-17930,'5,4':-770,
                 '5,5':-7400
                 }
            
            h = {'1,0':0.,'1,1':19730,
                 '2,0':0.,'2,1':-40410,'2,2':60300,
                 '3,0':0.,'3,1':-23100,'3,2': 51600,'3,3':-11310,
                 '4,0':0.,'4,1':32830,'4,2':-21310,'4,3':6060,'4,4':4860,
                 '5,0':0.,'5,1':20650,'5,2':-11670,'5,3':-2880,'5,4':-500,
                 '5,5':-22790
                 }            
        
        elif model == 'VIP4':
        
            order = 4
            g = {'1,0':420543., '1,1':-65920.,'2,0':-5118.,'2,1':-61904.,
                 '2,2':49690.,'3,0':-1576.,'3,1':-52036.,'3,2': 24386.,
                 '3,3':-17597.,'4,0':-16758.,'4,1': 22210.,'4,2':-6074.,
                 '4,3':-20243., '4,4': 6643.
                 }
            
            h = {'1,0':0, '1,1': 24992.,'2,0':0,'2,1':-36052., '2,2': 5250.,
                 '3,0':0, '3,1':-8804., '3,2': 40829.,'3,3':-31586.,'4,0':0,
                 '4,1': 7557.,'4,2': 40411.,'4,3':-16597.,'4,4': 3866. 
                 }
            
            
            
        elif model == 'VIT4':
            
            order = 4
            g = {'1,0':428077., '1,1':-75306,'2,0':-4283.,'2,1':-59426,
                 '2,2':44386,'3,0':8906.,'3,1':-21447.,'3,2': 21130.,
                 '3,3':-1190,'4,0':-22925,'4,1': 18940,'4,2':-3851,
                 '4,3':-9926, '4,4': 1271
                 }
            
            h = {'1,0':0, '1,1':  24616.,'2,0':0,'2,1':-50154., '2,2':  38452,
                 '3,0':0, '3,1':-17187., '3,2': 40667,'3,3':-35263,'4,0':0,
                 '4,1':  16088,'4,2': 11807,'4,3':6195,'4,4': 12641 
                 }
            
            
            
        elif model == 'Ulysses 17ev':
            
            order = 3
            g = {'1,0':410879,'1,1':-67885,
                 '2,0':7086.,'2,1':-64371.,'2,2':46437.,
                 '3,0':-5104.,'3,1':-15682.,'3,2':25148,'3,3':-15040,
                }
            
            h = {'1,0':0.,'1,1':22881,
                 '2,0':0.,'2,1':-30924,'2,2':13288,
                 '3,0':0.,'3,1':-15040,'3,2': 45743,'3,3':-21705,
                 }
            
        
               
        elif model == 'V1-17ev':
        
            order = 3
            g = {'1,0':420825,'1,1':-65980,
                 '2,0':-3411,'2,1':-75856,'2,2':48321,
                 '3,0':2153.,'3,1':-3295,'3,2':26315,'3,3':-6905,
                }
            
            h = {'1,0':0.,'1,1':26122,
                 '2,0':0.,'2,1':-29424,'2,2':10704,
                 '3,0':0.,'3,1':8883,'3,2': 69538,'3,3':-24718,
                 }
        
        
        
        elif model == 'O6':
            
            order = 3
            g = {'1,0':424200.0,'1,1':-65900.0,
                 '2,0':-2200.0,'2,1':-71100.0,'2,2':48700.0,
                 '3,0':7600.0,'3,1':-15500.0,'3,2':19800.0,'3,3':-18000.0,
                }
            
            h = {'1,0':0.,'1,1':24100.0,
                 '2,0':0.,'2,1':-40300.0,'2,2':7200.0,
                 '3,0':0.,'3,1':-38800.0,'3,2': 34200.0,'3,3':-22400.0,
                 }
        
        
        
        elif model == 'O4':
            
            order = 3
            g = {'1,0':421800,'1,1':-66400,
                 '2,0':-20300,'2,1':-73500,'2,2':51300,
                 '3,0':-23300,'3,1':-7600,'3,2':16800,'3,3':-23100,
                }
            
            h = {'1,0':0.,'1,1':26400,
                 '2,0':0.,'2,1':-46900,'2,2':8800,
                 '3,0':0.,'3,1':-58000,'3,2': 48700,'3,3':-29400,
                 }
          
            
            
        elif model == 'SHA':
            
            order = 3
            g = {'1,0':409200,'1,1':-70500,
                 '2,0':-3300,'2,1':-69900,'2,2':53700,
                 '3,0':-11300,'3,1':-58500,'3,2':28300,'3,3':6700,
                }
            
            h = {'1,0':0.,'1,1':23100,
                 '2,0':0.,'2,1':-53100,'2,2':7400,
                 '3,0':0.,'3,1':-42300,'3,2': 12000,'3,3':-17100,
                 }
            
        elif model == 'dipole':
            # This is a very simple dipole model!
            order = 1
            g = {'1,0':(430000), '1,1':0}
            h = {'1,0':0., '1,1':0}            
            
        else:
            print(' ======================================================= \n \
    ERROR: Please select an appropriate model: \n JRM09 \n VIP4 \n VIT4 \n \
Ulysses 17ev \n V1-17ev \n O6 \n O4 \n SHA \n \
========================================================')
            return None
        
        ##-------------------------------------------------------------------------
        
        
        P = self.SchmidtLegendre #Previously defined function to calculate the SQNL 
                            # functions. 
            
           
        dtheta = 1E-8*theta #used in the numerical differentiation of the SQNL
                            # functions
            
        a = 1 # radius of planet in Rj
            
            
                    
        ## create dictionaries to contain the field vectors which will later be
        ## summed. 
        Br = {}
        Bt = {}
        Bp = {}
        
        for i in range(1,order+1): # represents the order 
            m = range(0,i+1) # represents the degree
            Brterms = []
            Btterms = []
            Bpterms = []
            for j in m:
                leg_poly = P(i,j,theta)
                cosP = np.cos(j*phi)
                sinP = np.sin(j*phi)
                h_nm = h[str(i)+','+str(j)]
                g_nm = g[str(i)+','+str(j)]
                ##determines Br for fixed n, sum over all m
                ##for fixed n, and m
                Brterm = (i+1)*((a/r)**(i+2))*((g_nm*cosP)+\
                          (h_nm*sinP))*leg_poly
                
                Brterms.append(Brterm)
                ## sums all fixed n and m, producing array for Br n = 0 -> inf by m = 0 -> n
                Br[str(i)] = sum(Brterms) #sum this for internal Br
                
                ## same again for B theta and B phi
                Btterm = (-1)*((a/r)**(i+2))*((g_nm*cosP)+\
                          (h_nm*sinP))*\
                          ((P(i,j,theta+dtheta)-leg_poly)/dtheta)
                              
                Btterms.append(Btterm)
                Bt[str(i)] = sum(Btterms)
                    
                Bpterm = (1/np.sin(theta))*j*((a/r)**(i+2))*(g_nm*sinP-\
                                h_nm*cosP)*leg_poly
                    
                Bpterms.append(Bpterm)
                Bp[str(i)] = sum(Bpterms)
            
            
        Brint, Btint, Bpint = 0,0,0  
            
            ## Sum the individual values if n = ... m = 0 -> n giving internal field 
            ## at point r, theta, phi
        for k in range(1,len(Br)+1):
            Brint += Br[str(k)]
            Btint += Bt[str(k)]
            Bpint += Bp[str(k)]
                      
        
        return Brint, Btint, Bpint # returns the internal components, Br,Bt,Bp
    
    
    
        # ================================================
    