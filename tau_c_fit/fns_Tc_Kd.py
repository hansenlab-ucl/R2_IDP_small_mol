"""
  Class object and functions to fit chemical shift, R1_eff, R2_eff, and 
  DOSY data in support of 'Small-molecule binding to 
  intrinsically disordered proteins revealed by experimental NMR 19F 
  transverse spin-relaxation' by Heller, Shukla, Figueiredo, and Hansen.
  

  We assume a simple two-site exchange model between free (F) and bound (B) 
  5-fluoroindole:
  F<-->B
  where kforward  (-->) = kon * [P]
        kbackward (<--) = koff


  Useful references:
  1) Vallurupalli, Notes on Chemical Exchange, 2009.
  2) Cavanaugh, et al, Protein NMR Spectroscopy: Principles and Practice, 2006.
  3) Pages et al, Suppressing magnetization exchange effects in stimulated-echo 
     diffusion experiments, Journal of Magnetic Resonance, 2013.
  4) Bain and Duns, A unified approach to dynamic NMR based on a physical 
     interpretation of the transition probability, Canadian Journal of Chemistry, 1996.
  5) Lu et al, 19F Magic Angle Spinning NMR Spectroscopy and Density 
     Functional Theory Calculations of Fluorosubstituted Tryptophans: Integrating Experiment 
     and Theory for Accurate Determination of Chemical Shift Tensors, Journal of Physical 
     Chemistry B, 2018.
  6) Lu et al, 19F NMR relaxation studies of fluorosubstituted tryptophans, Journal of 
     Biomolecular NMR, 2020.
"""

# import useful tools
import os, sys
os.environ['MKL_NUM_THREADS']='1'       
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import scipy as sp
import scipy.linalg as sl
from scipy.linalg import expm
from scipy.optimize import least_squares
from pathos.multiprocessing import ProcessingPool as Pool

class fns_Tc_Kd:
    def __init__(self, nuc='1H', sfrq=500): #Default values are for a 500MHz spectrometer
        self.nuc=nuc
        self.sfrq=sfrq
        # set gyromagnetic ratios for various nuclei       
        self.gamma={}
        self.gamma['1H'] = 267.52218744e6
        self.gamma['19F']= 251.815e6
        self.gamma['13C']= 67.2828e6
        #
        # set parameters for the Diffusion Ordered SpectroscopY (DOSY) experiment
        self.grads = np.linspace(1, 20, 10, endpoint=True)/100 # T/m
        self.delta = 0.002  # unit in s
        self.Delta = 0.200  # unit in s
        #
        # set parameters for the nucleus
        self.params={}
        #
        # set parameters for e  xperimental data
        self.data={}
        #
        # set parameters for spectral information
        self.acqu={}
    #    
    # define a function to set acquisition times in a dictionary called 'acqu'
    def SetAcqu(self, param=None, value=None):
        self.acqu[param]=value
        if 'SW' in self.acqu.keys() and 'TD' in self.acqu.keys():
            # calculate the time increment based on the spectral width (SW) and 
            # the number of points (TD) 
            self.acqu['times']=np.arange(self.acqu['TD']//2)/self.acqu['SW']
        return
    #
    # define a function that takes values in ppm or Hz and converts them into radians per second    
    def SetParam(self, param=None, value=None):
        if isinstance(value,float): self.params[param]=value
        elif isinstance(value,str):
            # check for units such as 'ppm' or 'hz'
            if 'ppm' in value.lower():
                # If the value is in 'ppm', calculate the corresponding frequency in radians per 
                # second using the gyromagnetic ratio of the nucleus (self.gamma[self.nuc]), the 
                # gyromagnetic ratio of protons, the spectrometer frequency (self.sfrq), and 2*pi
                self.params[param]= \
                                    float(value.replace('ppm','')) * \
                                    (self.gamma[self.nuc]/self.gamma['1H']) * self.sfrq * \
                                    2. * np.pi
            if 'hz' in value.lower():
                # If the value is in 'hz', calculate the corresponding frequency in radians per 
                # second using the value, 'value' (in Hz), and 2*pi
                self.params[param]= \
                                    float(value.replace('hz','').replace('Hz','')) * 2. * np.pi
        else:
            # If the value is neither a float nor a string, print error
            print(f' Cannot set parameter {param} to value {value}', file=sys.stderr)
            sys.stderr.flush()
            sys.exit(10)

        return
    #
    # Define a longitudinal relaxation matrix to simulate and generate DOSY NMR spectra, 
    # under the influence of chemical exchange
    # Reference: Pages et al, Suppressing magnetization exchange effects in stimulated-echo 
    # diffusion experiments, Journal of Magnetic Resonance, 2013
    def GetGamma1(self, grad=None, kf=None, kb=None, Df=None, Db=None, R1f=None, R1b=None):
        # grad: gradient
        # kf: forward rate constant, also referred to as 'k_fb' in the supporting information. 
        # kb: backward rate constant, also referred to as 'k_bf' in the supporting information.
        # Df: Diffusion coefficient of free molecule
        # Db: Diffusion coefficient of bound molecule
        # R1f: R1 relaxation rate constant for the free molecule
        # R1b: R1 relaxation rate constant for the bound molecule
        if kf is None and kb is None and Df is None and Db is None:
            # Set parameters
            kf=self.params['kf'] # in s^-1
            kb=self.params['kb'] # in s^-1
            Df=self.params['Df'] # in m^2⋅s^-1
            Db=self.params['Db'] # in m^2⋅s^-1
            R1f=self.params['R1f'] # in s^-1
            R1b=self.params['R1b'] # in s^-1
        
        # set parameters
        if R1f is None and R1b is None:
            R1f=self.params['R1f'] # in s^-1
            R1b=self.params['R1b'] # in s^-1
        #
        G = np.zeros( (2,2) )
        q = self.gamma[self.nuc] * grad * self.delta/(np.pi * 2)
        #
        G[0,0] = -np.square( 2 * np.pi * q )*Df - kf - R1f
        G[1,0] = kf
        G[0,1] = kb
        G[1,1] = -np.square( 2 * np.pi * q )*Db - kb - R1b
        #
        return G
    #
    # define a transverse relaxation matrix
    def GetGamma2(self, kf=None, kb=None, R2f=None, R2b=None, DeltaOmega=None):
        #
        if kf is None and kb is None and DeltaOmega is None:
            # set parameters
            kf=self.params['kf']    # in s^-1
            kb=self.params['kb']    # in s^-1
            DeltaOmega=self.params['DeltaOmega'] # in rad/sec
        
        if R2f is None and R2b is None:
            R2f=self.params['R2f']  # in s^-1
            R2b=self.params['R2b']  # in s^-1
        #
        G = np.zeros( (2,2), dtype=np.complex128 )    
        G[0,0] = -kf - R2f
        G[1,0] = kf
        G[0,1] = kb
        G[1,1] = -kb - R2b + 1j*DeltaOmega
        #
        return G
    #
    # define a function for the longitudinal relaxation matrix
    def GetGamma3(self, grad=None, kf=None, kb=None, Df=None, Db=None, R1f=None, R1b=None):
        #
        # returns the longitudinal relaxation matrix with identify 
        #
        if kf is None and kb is None and Df is None and Db is None:
            # set parameters
            kf=self.params['kf'] #s^-1
            kb=self.params['kb'] #s^-1
        # set parameters
        if R1f is None and R1b is None:
            R1f=self.params['R1f'] #s^-1
            R1b=self.params['R1b'] #s^-1

        #
        G = np.zeros( (3,3) )
        #
        G[0,0] = 0 
        G[0,1] = 0
        G[0,2] = 0
        G[1,0] = R1f*(kb/(kf + kb))
        G[2,0] = R1b*(kf/(kf + kb))
        G[1,1] = - kf - R1f
        G[2,1] = kf
        G[1,2] = kb
        G[2,2] = - kb - R1b

        return G
    #
    # Define a set of functions that takes in R1 (longitudinal) and R2 (transverse) relaxation 
    # rates and returns a rotational correlation time, tc.
    #
    # Equations come from Cavanaugh, et al, Protein NMR Spectroscopy: Principles and Practice, 2006
    # accounding for the dipole-dipole interactions of fluorine with the nearest hydrogen 
    # and the 19F chemical shift anisotropy (CSA) contribution.
    #
    # Default values for principal components of the chemical shift tensor (CST) for 5-fluoroindole  
    # were taken from 5-DL-tryptophan as reported in Lu et al, 19F Magic Angle Spinning NMR  
    # Spectroscopy and Density Functional Theory Calculations of Fluorosubstituted Tryptophans:  
    # Integrating Experiment and Theory for Accurate Determination of Chemical Shift Tensors,  
    # Journal of Physical Chemistry B, 2018.
    #
    # Default values for distance between 19F and all 1H atoms within 3A was taken from Lu et al, 19F NMR 
    # relaxation studies of fluorosubstituted tryptophans, Journal of Biomolecular NMR, 2020. This 
    # includes the 1_H_epsilon_3 and 1_H_eta_2 atoms which are equidistant from 19F at 2.6A. 
    #
    def R1_R2_tau_c(self, 
                    dxx=-86.1, # principal component CST in ppm,  val ref: Lu et al, J. Phys. Chem. B, 2018.
                    dyy=-60.5, # principal component CST in ppm,  val ref: Lu et al, J. Phys. Chem. B, 2018.
                    dzz=4.8,   # principal component CST in ppm, val ref: Lu et al, J. Phys. Chem. B, 2018.
                    b0=11.7,   # 500 MHz
                    m0=1.2566*10**-6, # vacuum magnetic permeability constant in N⋅A^-2
                    h=6.62607015*10**(-34), # planck's constant, in J⋅Hz^-1
                    gf=251.8*10**6, # gyromagnetic ratio of 19F in rad⋅s^−1⋅T^−1
                    gh=267.5*10**6, # gyromagnetic ratio of 1H in rad⋅s^−1⋅T^−1
                    wf_val = 2*np.pi*470*10**6, # resonance frequency of 19F at 500MHz, in rad⋅s^-1 
                    wh_val = 2*np.pi*500*10**6, # resonance frequency of 1H at 500MHz, in rad⋅s^-1
                    rfh=2.6e-10, # distance between 19F and nearest 1H, in m. val ref: Lu et al, JBNMR,2020.
                    ds=52.1, # delta sigma = delta_zz - delta_iso, in ppm.
                    tc=0.013, # rotational correlation time, tau_c, in ns.
                    d=None, # spatial function for dipolar relaxation, see definition below
                    Jwf=None, # spectral densitry function J(w_19F)
                    Jwh=None, # spectral densitry function J(w_1H)
                    J0=None, # spectral densitry function J(0)
                    Jwh_m_f=None, # spectral densitry function J(w_1H-w_19F)
                    Jwf_p_h=None, # spectral densitry function J(w_1H+w_19F)
                    c=None, # spatial function for CSA relaxation, see definition below
                    ten=None): # CSA asymmetry parameter
        #
        #define the isotropic chemical shift
        def delta_iso(dxx=dxx, dyy=dyy, dzz=dzz):
            return ((1/3)*(dxx+dyy+dzz))
        #
        # define the reduced anisotropy of the 19F chemical shift tensor
        def delta_sigma(dzz=dzz, diso=None):
            if diso is None:
                diso = delta_iso()
            return (dzz-diso)
        #
        # define the asymmetry parameter of the 19F chemical shift anisotropy tensor
        def tensor_n(dyy=dyy, dxx=dxx, dsig=None):
            if dsig is None:
                dsig = delta_sigma()
            return ((dyy-dxx)/dsig)
        #
        # define the spatial function for dipolar relaxation mechanism
        def d_AP(m0=m0, h=h, gh=gh, gf=gf, rfh=rfh):
            return (m0*h*gh*gf)/(8.*(np.pi)**2*rfh**3.)
        #
        # define the spatial function for the CSA relaxation mechanism
        def c_AP(gf=gf, b0=b0, ds=ds):
            if ds is None:
                ds = delta_sigma()
            return (gf*b0*ds*10**-6)/(np.sqrt(3.))
        #
        # define the spectral densitry function
        def J_AP(tc=tc, w=None):
            return (2./5.)*((tc*1e-9)/(1+w**2*(tc*1e-9)**2))

        if d is None:
            d = d_AP()
        if Jwh is None:
            Jwh = J_AP(w=wh_val)
        if Jwf is None:
            Jwf = J_AP(w=wf_val)
        if Jwh_m_f is None:
            Jwh_m_f = J_AP(w=wh_val-wf_val)
        if Jwf_p_h is None:
            Jwf_p_h = J_AP(w=wf_val+wh_val)
        if J0 is None:
            J0 = J_AP(w=0)
        if c is None:
            c = c_AP()
        if ten is None:
            ten = tensor_n()
        
        # define R1 and R2 based on tau_c (tc)
        # note that the factor of 2 in the dipolar relaxation term is to account for the
        # 2 equidistant 1H (see above)
        R1_tauc = (d**2./4.)*2*(3.*Jwf+Jwh_m_f+6.*Jwf_p_h)+c**2.*Jwf
        R2_tauc = (d**2./8.)*2.*(4.*J0+3.*Jwf+Jwh_m_f+6.*Jwh+6.*Jwf_p_h)+(c**2/6.)*(4.*J0+3.*Jwf)

        return R1_tauc, R2_tauc
    #
    # define function to read in data 
    def ReadData(self, datatype=None, columns=(0,1,2), filename=None ):
        # columns = [x, y, err]
        self.data[datatype]=[]
        with open(filename,'r') as f:
            for l in f.readlines():
                if l[0]=='#': continue
                if len(l.split())<2: continue
                its = l.split()
                #
                try:
                    self.data[datatype].append([ float(its[i]) for i in columns ])
                except(ValueError):
                    continue
        self.data[datatype]=np.array(self.data[datatype])   
        return
    #
    # define a function to calculate the effective R2 (transverse) relaxation rates and 
    # omega values for 5-fluoroindoe given concentrations of protein (pconcs), tauCPMG 
    # (CPMG delay), number of CPMG blocks (ncyc), R2 of free molecule (R2F) and R2 of 
    # bound molecule (R2b), using the GetGamma2 matrix above.
    def CalcTransverse(self, pconcs=None, tauCPMG=None, ncyc=(1,7,13,19,25,30), R2b=None, R2f=None ):
        #
        # Return effective R2 rates and omega values
        #
        if isinstance(pconcs,float):
            pconcs=np.array([pconcs])
        else:
            pconcs=np.array(pconcs)

        R2=[] # define empty list for R2_eff values
        Om=[] # define empty list for Omega_eff values
        for p in pconcs:
            # update kf and kb values based on the given protein concentration
            self.params['kf'] = self.params['kon'] * p
            self.params['kb'] = self.params['koff']

            if R2f is None and R2b is None:
                R2f=self.params['R2f']
                R2b=self.params['R2b']
            #
            # get the Gamma2 matrix and calculate its eigenvalues
            G = self.GetGamma2(R2b=R2b, R2f=R2f)
            ev = np.linalg.eigvals( G )
            #
            # find the index of the smallest negative real part of eigenvalue
            idx = np.argmin( -ev.real )
            # observed chemical shift, omega_eff, is the imaginary part of this eigenvalue
            Om.append(  ev[idx].imag )
            #
            if tauCPMG is None:
                # the R2_eff relaxation rate is the negative real part of this eigenvale
                R2.append( -ev[idx].real )
            else:
                # simulate the CPMG-based R2_eff experiment and fit rate
                #
                # calculate the initial population values, based on definitons of K_d
                initp =np.array([\
                                 self.params['kb']/(self.params['kf'] + self.params['kb']),\
                                 self.params['kf']/(self.params['kf'] + self.params['kb']) \
                ])
                # define propagator for the CPMG block (tau-pi-2*tau-pi-tau)
                Prop = expm( tauCPMG * G) @ np.conj( expm( 2 * tauCPMG * G ) ) @ expm( tauCPMG * G ) 
                # calculate intensities for all values of number of cycles (ncyc)
                Ints=[]
                for n in ncyc:
                    # apply propagator n times from ncyc
                    Ints.append( np.sum(np.linalg.matrix_power(Prop, n) @ initp ).real  )
                Ints=np.array(Ints)
                #
                # fit exponential to calculate R2_eff value
                def f2min(params):
                    #params[0]: Intensity
                    #params[1]: R2eff
                    calc = params[0]*np.exp( - 4 * tauCPMG * np.array(ncyc) * params[1] )
                    return calc - Ints
                
                res = least_squares( f2min, [1, -ev[idx].real], verbose=0)
                R2.append( res.x[1] )
                
        return np.array(Om), np.array(R2)
    #
    # define a function to generate DOSY plots, based on a single protien concentration, 
    # and the R1 and R2 relaxtaion rates for free and boud molecule
    def GenerateDOSY(self, pconc=None, grads=None, R2f=None, R2b=None, R1f=None, R1b=None):
        # this only works for a single concentration of protein
        if not isinstance(pconc,float):
            print(f' .GenerateDOSY() only works for a single protein concentration', file=sys.stderr)
            sys.stderr.flush()
            sys.exit(10)
        # 
        if grads is None:
            grads = self.grads
        #
        # calculate forward and backward rate constants
        self.params['kf'] = self.params['kon'] * pconc
        self.params['kb'] = self.params['koff']    
        #
        # calculate the initial populations at equilibrium
        initp =np.array([\
                         self.params['kb']/(self.params['kf'] + self.params['kb']),\
                         self.params['kf']/(self.params['kf'] + self.params['kb']) \
                         ])
        #
        grads = np.array(grads)
        # if 'grads' has more than one dimension, print error
        if len(grads.shape)>1:
            print(f' Gradients must be a 1D array ', file=sys.stderr)
            sys.stderr.flush()
            sys.exit(10)
        #
        # allocate memory for full spectrum
        Output = np.zeros( (self.acqu['times'].shape[-1], grads.shape[-1]))
        #
        # set window function
        window = np.square( np.cos( 0.5*np.pi * np.arange(self.acqu['times'].shape[-1])/self.acqu['times'].shape[-1]))
        window[0] *=0.5
        #
        # loop over each gradient value in 'grads'
        for g in range(grads.shape[-1]):
            #
            # calculate the intensities due to diffusion
            Ints = expm(self.Delta * self.GetGamma1(grads[g],R1f=R1f, R1b=R1b)) @ initp
            #
            # apply 90 pulse and acquire the FID
            Prob = expm( (1./self.acqu['SW']) * self.GetGamma2(R2f=R2f, R2b=R2b) ) # Get propagator
            nFID = [Ints]
            for i in range(self.acqu['times'].shape[-1]-1): nFID.append( Prob @ nFID[-1] )
            nFID = np.array(nFID)
            FID = np.sum(nFID, axis=-1)
            #
            # perform Fourier transform and store the real part of the resulting spectrum
            spec = np.fft.fftshift( np.fft.fft( FID * window))
            Output[:,g] = np.copy( spec.real )
        #
        # calculate frequency axis for the output spectrum
        freqs = np.linspace(-self.acqu['SW']/2.,self.acqu['SW']/2., self.acqu['times'].shape[-1] + 1)
        freqs = freqs[:-1]

        return Output, freqs
    # define a function to calculate the effective R1 (longitudinal) relaxation rates for 
    # 5-fluoroindoe given concentrations of protein (pconcs), variable delay lists (vdlist) 
    # R1 of free molecule (R1F) and R1 of bound molecule (R1b), using the GetGamma3 matrix above.
    def CalcLongitudinal(self, pconcs=None, vdlist=(.078125,.15625,.3125,.625,1.25,2.5,5,10,20), R1f=None, R1b=None ):
        #
        # Return effective R1 rates
        #
        if isinstance(pconcs,float):
            pconcs=np.array([pconcs])
        else:
            pconcs=np.array(pconcs)

        R1=[] # define empty list for R1_eff values
        for p in pconcs:
            # update kf and kb values based on the given protein concentration
            self.params['kf'] = self.params['kon'] * p
            self.params['kb'] = self.params['koff']
            #
            # get equilibrium populations
            initp =np.array([\
                1,\
                -self.params['kb']/(self.params['kf'] + self.params['kb']),\
                -self.params['kf']/(self.params['kf'] + self.params['kb']) \
                    ])

            # get the Gamma3 matrix 
            G = self.GetGamma3(R1f=R1f, R1b=R1b)

            Ints=[]
            for v in vdlist:
                # Propagrator for T1 inversion recovery experiment
                Prop = expm( v * G ) 
                Ints.append( np.asarray(Prop @ initp)[-2:].sum() )
            Ints=np.array(Ints)
            # fit rate based on exponential funciton
            def f2min(params_test):
                # params_test[0]: M1_eq
                # params_test[1]: R1
                # params_test[2]: M1_t0 
                calc = (params_test[2] - params_test[0])*(np.exp(-params_test[1]*np.array(vdlist)))+params_test[0]
                return calc - Ints

            res = least_squares( f2min, [1, .2, -1], verbose=0)
            R1.append( res.x[1] )
        return np.array(R1)
    #    
    # define function to get effective diffusion coefficient 
    def GetDeff(self,dosy):
        peak_idx = np.argmax( dosy[:,0] )
        #
        # fit the diffusion equation
        def f2min(params):
            # params[0]: intensity
            # params[1]: diffusion
            calc = params[0] * np.exp( -np.square( self.gamma[self.nuc] * self.delta * self.grads)*(self.Delta-self.delta/3.) * params[1])
            return calc - dosy[peak_idx,:]/np.max(dosy[peak_idx,:])

        lb = (0.9, self.params['Db']*0.9)
        ub = (1.1, self.params['Df']*1.1)
    
        res = least_squares( f2min, [1., self.params['Df'] ], \
                             bounds = ( lb, ub ), verbose=0)
    
        return res.x[1]
    
