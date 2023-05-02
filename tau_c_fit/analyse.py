#!/usr/bin/env python3

"""

  Analysis of 5FI binding to the disordered protein, NS5A-D2D3.  
  Analyse chemical shift, R1_eff, R2_eff, and DOSY data in support 
  of 'Small-molecule binding to intrinsically disordered proteins revealed 
  by experimental NMR 19F transverse spin-relaxation' by Heller, Shukla, 
  Figueiredo, and Hansen.
 
"""
# import useful tools
import os,sys
os.environ['MKL_NUM_THREADS']='1'       
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy import linalg
from scipy.optimize import least_squares
from pathos.multiprocessing import ProcessingPool as Pool
from fns_Tc_Kd import * #requires class object & functions from 'fns_Tc_Kd.py'

# set useful parameters for the DOSY fit
FI5=fns_Tc_Kd(nuc='1H') # set nucleus for DOSY fit
FI5.grads =  np.linspace(1, 20, 10, endpoint=True)/100 # set gradients 
FI5.delta =  0.002 # set little delta, in s
FI5.Delta =  0.2   # set big delta, in s
FI5.SetAcqu('SW',2000.) # set acquisition time 
FI5.SetAcqu('TD',1024)  # acqt = (TD/2) / SW

#
#Set (initial) parameters
FI5.SetParam('R1f',0.17332563) # set initial parameter for R1_free, value from IR expt in s-1
FI5.SetParam('R1b',0.21) # set initial parameter for R1_bound in s-1
FI5.SetParam('R2f',0.181806087) # set initial parameter for R2_free, value from CPMG_R2eff expt in s-1
FI5.SetParam('R2b',0.80) # set initial parameter for R2_bound in s-1
FI5.SetParam('DeltaOmega','0.05ppm') # set initial parameter for DeltaOmega
FI5.SetParam('Df', 1.524e-9) # set initial parameter for diffusion coeff of free mol, value from DOSY expt in m^2/s
FI5.SetParam('Db', 1.05e-10) # set initial parameter for diffusion coeff of bound mol in m^2/s
FI5.SetParam('kon', 3.3e6) # set initial parameter for k_on rate in M-1s−1
FI5.SetParam('koff',400.) # set initial parameter for k_off rate in s-1
FI5.SetParam('tcf', 0.02057741) # set initial parameter for tau_c_free in ns
FI5.SetParam('tcb', 2.) # set initial parameter for tau_c_bound in ns
#
# read in data and errors from CPMG-R2eff experiment
FI5.ReadData(datatype='R2', columns=(0,1,2), filename='data.inp')
FI5.data['R2'][:,0]*=1e-6 # convert uM to M
#
# read in chemical shift data and errors 
FI5.ReadData(datatype='CS', columns=(0,3,4), filename='data.inp')
FI5.data['CS'][:,0]*=1e-6 # convert uM to M
FI5.data['CS'][:,1]=(FI5.data['CS'][:,1]-FI5.data['CS'][0,1])* \
                     (FI5.gamma[FI5.nuc]/FI5.gamma['1H']) * FI5.sfrq * 2. * np.pi  # convert to rad/sec
FI5.data['CS'][:,2]=(FI5.data['CS'][:,2])* \
                     (FI5.gamma[FI5.nuc]/FI5.gamma['1H']) * FI5.sfrq * 2. * np.pi  # convert to rad/sec

# read in data and errors from inversion recovery (R1eff) experiment
FI5.ReadData(datatype='R1', columns=(0,5,6), filename='data.inp')
FI5.data['R1'][:,0]*=1e-6 # convert uM to M


# define fitting function for koff, tcf, tbc, omega
def Fitting(param_names, lb=None, ub=None, verbose=0):
    #
    #print(' Now fitting ',param_names)

    def f2min(params):
        for i,p in enumerate(param_names):
            # if param is chemical shift, give 18 decimal places
            if 'omega' in p.lower():
                FI5.SetParam(p,'%.18fppm' %(params[i],))
            else:
                FI5.SetParam(p,params[i])
                # print('setting', p, params[i]) #print paramater and value

        # calculate R1_free and R2_free from tau_c of free molecule
        R1f_tauc, R2f_tauc = FI5.R1_R2_tau_c(tc= FI5.params['tcf'])
        # calculate R1_bound and R2_bound from tau_c of bound molecule
        R1b_tauc, R2b_tauc = FI5.R1_R2_tau_c(tc= FI5.params['tcb'])
        FI5.SetParam('R1f',R1f_tauc) # set new value of R1_bound
        FI5.SetParam('R2f',R2f_tauc) # set new value of R2_bound
        FI5.SetParam('R1b',R1b_tauc) # set new value of R1_bound
        FI5.SetParam('R2b',R2b_tauc) # set new value of R2_bound
        #
        # calculate chemical shifts and R2_eff
        _ , R2eff   = FI5.CalcTransverse(pconcs=FI5.data['R2'][:,0], tauCPMG=0.01, R2b=R2b_tauc, R2f=R2f_tauc)
        Omegaeff ,_ = FI5.CalcTransverse(pconcs=FI5.data['CS'][:,0], R2b=R2b_tauc, R2f=R2f_tauc)  
        #
        # calculate effective diffusion coefficient  
        dosy, freqs = FI5.GenerateDOSY(pconc= 75e-6,  R2f=R2f_tauc, R2b=R2b_tauc, R1f=R1f_tauc, R1b=R1b_tauc)
        Deff = FI5.GetDeff(dosy)
        #
        # calculate R1_eff
        R1calc = FI5.CalcLongitudinal(pconcs=FI5.data['R1'][:,0], R1f=R1f_tauc, R1b=R1b_tauc)

        calc = [] # empty list for calculated observables
        obs  = [] # empty list for experimentally measured observables
        esd  = [] # empty list for experimentally measured errors
        #
        # add R2_eff to chi-squared
        for i in range(FI5.data['R2'][:,0].shape[-1]):
            calc.append( R2eff[i] )
            obs.append( FI5.data['R2'][i,1] )
            esd.append( FI5.data['R2'][i,2] )
        # add CS to chi-squared
        for i in range(FI5.data['CS'][:,0].shape[-1]): 
            calc.append( Omegaeff[i] ) 
            obs.append( FI5.data['CS'][i,1] )
            esd.append( FI5.data['CS'][i,2] )
        #
        # add diffusion to chi-squared
        calc.append( Deff )
        obs.append( FI5.params['Df'] )
        esd.append( 0.08*FI5.params['Df'] )  # 8% error

    # add R1_eff to chi-squared
        for i in range(FI5.data['R1'][:,0].shape[-1]): 
            calc.append( R1calc[i] ) 
            obs.append( FI5.data['R1'][i,1] )
            esd.append( FI5.data['R1'][i,2] )             
        # calculate weighted least squares
        return (np.array(calc) - np.array(obs))/np.array(esd)
    #
    #set initial parameters
    initp=[]
    for i,p in enumerate(param_names):
        if 'omega' in p.lower():
            # scale CS by nucleus w.r.t. 1H
            ScalingFactor = 2 * np.pi * FI5.sfrq * FI5.gamma[FI5.nuc] / FI5.gamma['1H']
            initp.append( FI5.params[p]/ScalingFactor )
            FI5.SetParam(p,'%.18fppm' %(FI5.params[p],))
        else:
            initp.append(FI5.params[p])
    #       
    # do least squares fitting
    # if no bounds given:
    if lb is None and ub is None:
        res = least_squares( f2min, initp, verbose )
    # with bounds given
    else:
        res = least_squares( f2min, initp, bounds=(lb, ub), verbose=0 )        
    #
    # calculate uncertainties and covariance from the Jacobian
    U, S, Vh = linalg.svd( res.jac, full_matrices=False) # SVD of Jacobian
    tol = np.finfo(float).eps*S[0]*max(res.jac.shape) # calculate tolerance, singular vals significance
    w = S > tol # determine which values are significant
    #
    # compute estimated covariance matrix
    cov = (Vh[w].T/S[w]**2) @ Vh[w]
    #
    # normalize
    redchi2 = np.sum( np.square( res.fun))/(res.fun.size - res.x.size ) #reduced chi-squared
    cov *= redchi2 # scale estimated covariance matrix by reduced chi-squared
    std = np.sqrt(np.diag(cov)) # compute standard deviations of params
    #
    # get correlation matrix
    Dinv = np.diag( 1./ std )
    corr = Dinv @ cov @ Dinv # pairwise correlation coefficients between params
    #
    if verbose>0:
        print(f'# Chi-squared:     {res.cost*2.}' )
        print(f'# Red-Chi-squared: {redchi2 }')
    # print optimized params and uncertainties
    names=param_names 
    for i in range(res.x.size):
        if verbose>0: print(f'# {names[i] :12s} {res.x[i] :13.6e} +/- {std[i] :13.6e}')
    #
    # print correlation matrix
    if verbose>0:
        print('#\n# Correlation Matrix ')
        for i in range(res.x.size):
            print('#', end='')
            for j in range(res.x.size):
                print(f'{ corr[i,j]  :6.3f} ',end='')
            print('')
    #
    return res.x, std, corr, res

# define fitting function for kd, koff, tcf, tbc, and omega, 
def FittingKD(param_names, lb=None, ub=None, verbose=0):
    #
    def f2min(params):
        # params[0] = kd 
        # params[1] = koff
        # params[2] = DeltaOmega
        # params[3] = tcf 
        # params[4] = tcb 
        for i,p in enumerate(param_names):
            # if param is chemical shift, give 18 decimal places
            if 'omega' in p.lower():
                FI5.SetParam(p,'%.18fppm' %(params[i],))
            else:
                FI5.SetParam(p,params[i])
                # print('setting', p, params[i]) #print paramater and value
        #
        if 'kd' in param_names:
            FI5.SetParam('kon', params[1]/params[0])
            FI5.SetParam('koff',params[1])
            FI5.SetParam('kd',params[0])
        
        # calculate R1_free and R2_free from tau_c of free molecule
        R1f_tauc, R2f_tauc = FI5.R1_R2_tau_c(tc= FI5.params['tcf'])
        # calculate R1_bound and R2_bound from tau_c of bound molecule
        R1b_tauc, R2b_tauc = FI5.R1_R2_tau_c(tc= FI5.params['tcb'])
        FI5.SetParam('R1f',R1f_tauc) # set new value of R1_bound
        FI5.SetParam('R2f',R2f_tauc) # set new value of R2_bound
        FI5.SetParam('R1b',R1b_tauc) # set new value of R1_bound
        FI5.SetParam('R2b',R2b_tauc) # set new value of R2_bound
        #
        # calculate chemical shifts and R2_eff
        _ , R2eff   = FI5.CalcTransverse(pconcs=FI5.data['R2'][:,0], tauCPMG=0.01, R2b=R2b_tauc, R2f=R2f_tauc)
        Omegaeff ,_ = FI5.CalcTransverse(pconcs=FI5.data['CS'][:,0], R2b=R2b_tauc, R2f=R2f_tauc)  
        #
        # calculate effective diffusion coefficient  
        dosy, freqs = FI5.GenerateDOSY(pconc= 75e-6,  R2f=R2f_tauc, R2b=R2b_tauc, R1f=R1f_tauc, R1b=R1b_tauc)
        Deff = FI5.GetDeff(dosy)
        #
        # calculate R1_eff
        R1calc = FI5.CalcLongitudinal(pconcs=FI5.data['R1'][:,0], R1f=R1f_tauc, R1b=R1b_tauc)

        calc = [] # empty list for calculated observables
        obs  = [] # empty list for experimentally measured observables
        esd  = [] # empty list for experimentally measured errors
        #
        # add R2_eff to chi-squared
        for i in range(FI5.data['R2'][:,0].shape[-1]):
            calc.append( R2eff[i] )
            obs.append( FI5.data['R2'][i,1] )
            esd.append( FI5.data['R2'][i,2] )
        # add CS to chi-squared
        for i in range(FI5.data['CS'][:,0].shape[-1]): 
            calc.append( Omegaeff[i] ) 
            obs.append( FI5.data['CS'][i,1] )
            esd.append( FI5.data['CS'][i,2] )
        #
        # add diffusion to chi-squared
        calc.append( Deff )
        obs.append( FI5.params['Df'] )
        esd.append( 0.08*FI5.params['Df'] )  # 8% error

	# add R1_eff to chi-squared
        for i in range(FI5.data['R1'][:,0].shape[-1]): 
            calc.append( R1calc[i] ) 
            obs.append( FI5.data['R1'][i,1] )
            esd.append( FI5.data['R1'][i,2] )             
        # calculate weighted least squares
        return (np.array(calc) - np.array(obs))/np.array(esd)
    #
    # set initial parameters
    initp=[]
    if 'kd' in param_names:
        FI5.SetParam('kd',FI5.params['koff']/FI5.params['kon'])

    for i,p in enumerate(param_names):
        if 'omega' in p.lower():
            # scale CS by nucleus w.r.t. 1H
            ScalingFactor = 2 * np.pi * FI5.sfrq * FI5.gamma[FI5.nuc] / FI5.gamma['1H']
            initp.append( FI5.params[p]/ScalingFactor )
            FI5.SetParam(p,'%.18fppm' %(FI5.params[p],))
        else:
            initp.append(FI5.params[p])
    #       
    # do least squares fitting
    # if no bounds given:
    if lb is None and ub is None:
        res = least_squares( f2min, initp, verbose )
    # with bounds given
    else:
        res = least_squares( f2min, initp, bounds=(lb, ub), verbose=0 )        
    #
    # calculate uncertainties and covariance from the Jacobian
    U, S, Vh = linalg.svd( res.jac, full_matrices=False) # SVD of Jacobian
    tol = np.finfo(float).eps*S[0]*max(res.jac.shape) # calculate tolerance, singular vals significance
    w = S > tol # determine which values are significant
    #
    # compute estimated covariance matrix
    cov = (Vh[w].T/S[w]**2) @ Vh[w]
    #
    # normalize
    redchi2 = np.sum( np.square( res.fun))/(res.fun.size - res.x.size ) #reduced chi-squared
    cov *= redchi2 # scale estimated covariance matrix by reduced chi-squared
    std = np.sqrt(np.diag(cov)) # compute standard deviations of params
    #
    # get correlation matrix
    Dinv = np.diag( 1./ std )
    corr = Dinv @ cov @ Dinv # pairwise correlation coefficients between params
    #
    if verbose>0:
        print(f'# Chi-squared:     {res.cost*2.}' )
        print(f'# Red-Chi-squared: {redchi2 }')
    # print optimized params and uncertainties
    names=param_names 
    for i in range(res.x.size):
        if verbose>0: print(f'# {names[i] :12s} {res.x[i] :13.6e} +/- {std[i] :13.6e}')
    #
    # print correlation matrix
    if verbose>0:
        print('#\n# Correlation Matrix ')
        for i in range(res.x.size):
            print('#', end='')
            for j in range(res.x.size):
                print(f'{ corr[i,j]  :6.3f} ',end='')
            print('')
    #
    return res.x, std, corr, res

# Fit
print(f' ====== First fitting with kon = 3.3e6 M-1s-1 ====== ') # kon set in initial parameters above
val, err, corr, res = Fitting(['koff','DeltaOmega','tcf','tcb'], lb=(0.0001, -0.1, 0.01,0.01), ub=(1e6, 0.1, 1000.0,1000.0), verbose=1)
print(f' ================================================= ')
#
print(' Running chi-squared surface ... ', end='')
sys.stdout.flush()
#
# do a chi-squared surface of Kd and koff (using all cores)
# define arrays of K_ds and k_off values for chi-squared surface
kds =  1e-4 * np.power(10, np.linspace(0,1.5, 21, endpoint=True) )  # 0.1 mM to 10 mM

koffs= np.linspace(400,8000,21,endpoint=True)                       #

# define chi-squared function
def GetChi2(kd,koff):

    FI5.SetParam('kon', koff/kd)
    FI5.SetParam('koff',koff)

    # store values and put back later
    tcf = FI5.params['tcf']
    tcb = FI5.params['tcb']
    DO  = FI5.params['DeltaOmega']
    # 
    val, err, corr, res = Fitting(['DeltaOmega','tcf','tcb'], lb=(-0.1,0.01,0.01), ub=(0.1,1000.0,1000), verbose=0)
    FI5.params['tcf']=tcf
    FI5.params['tcb']=tcb
    FI5.params['DeltaOmega']=DO

    return res.cost * 2.    # factor of 2 converts sum of squared residuals to the chi-squared.

# initialize empty lists to store kd and koff vals
x_list=[]
y_list=[]
# loop over each kd and koff val and append to respective lists
for kd in kds:
    for koff in koffs:
        x_list.append(kd)
        y_list.append(koff)
# use multiprocessing module to parallelize GetChi2 for each pair of kd and koff
with Pool( os.cpu_count() ) as p: # os.cpu_count() returns the number of available CPUs
    Chi2 = p.map(GetChi2, x_list, y_list) # map GetChi2 to each pair of kd and koff

Chi2=np.array(Chi2)

best_idx = np.argmin( Chi2 ) # find the index of min chi-squared val

# make figure of chi-squared surface
# create plot
fig, ax = plt.subplots()
ax.imshow( np.exp( -(Chi2.reshape(21,21).T-np.min(Chi2))/2. ), \
            cmap='hot_r', \
            extent=(np.min(np.log10(kds)),np.max(np.log10(kds)),np.min(koffs),np.max(koffs)), \
            aspect='auto', \
            origin='lower',
            interpolation='quadric')


ax.contour( Chi2.reshape(21,21).T-np.min(Chi2), \
             origin='lower', \
             extent=(np.min(np.log10(kds)),np.max(np.log10(kds)),np.min(koffs),np.max(koffs)), \
             colors=['k','k'], linestyles=['solid','dashed'], \
             levels=[2.30, 6.18] ) # For two variables 1 sigma at 2.30, 2 sigma at 6.18

ax.set_xlabel(r' log$_{10}({\rm K}_{\rm d})$')
ax.set_ylabel(r' $k_{\rm off}$  s$^{-1}$')
fig.savefig('chi2surface.png',  dpi=300)
# return figure object
plt.clf()
plt.close()

print(' DONE ')
sys.stdout.flush()

print(f' ====== Final fitting with kon from minimum ====== ')
print(f' kon = { y_list[best_idx]/x_list[best_idx] :13.6e} ') # print calculated kon
print(f' koff = { y_list[best_idx] :13.6e} ') # print calculated kon
print(f' kd = { x_list[best_idx] :13.6e} ') # print calculated kon

sys.stdout.flush() # flush the standard output buffer

# set value of koff, kon, and kD where chi-squared val is a minimum
FI5.SetParam('koff', y_list[best_idx])
FI5.SetParam('kon' , y_list[best_idx]/x_list[best_idx])
# calculate  R1f_tauc, R2f_tauc, R1b_tauc, and R2b_tauc from tau_c_free and tau_c_bound
R1f_tauc, R2f_tauc = FI5.R1_R2_tau_c(tc=FI5.params['tcf'])
R1b_tauc, R2b_tauc = FI5.R1_R2_tau_c(tc=FI5.params['tcb'])
# do fit
val, err, corr, res = FittingKD(['kd','koff','DeltaOmega','tcf','tcb'], lb=(0.0,0.0, -0.1, 0.01,0.01), ub=(1,1e6, 0.1, 1000.0,1000.0), verbose=1)
# back-calculate experimental data
_ , R2eff   = FI5.CalcTransverse(pconcs=FI5.data['R2'][:,0], tauCPMG=0.01, R2b=R2b_tauc, R2f=R2f_tauc)
Omegaeff ,_ = FI5.CalcTransverse(pconcs=FI5.data['CS'][:,0], R2b=R2b_tauc, R2f=R2f_tauc)
R1calc      = FI5.CalcLongitudinal(pconcs=FI5.data['R1'][:,0], R1b=R1b_tauc, R1f=R1f_tauc)

# make a figure comparing predicted data to experimental data 
fig, axs = plt.subplots(3, 1, figsize=(5, 9.2), sharex=True)
plt.subplots_adjust(hspace=0.4)

axs[0].errorbar(FI5.data['R2'][:,0]*1e6, FI5.data['R2'][:,1], yerr=FI5.data['R2'][:,2], c='r', marker='o', linestyle='' )
axs[0].plot(FI5.data['R2'][:,0]*1e6, R2eff, 'r--')
#axs[0].set_xlabel(r' [NS5A]  ($\mu$M)')
axs[0].set_ylabel(r' $R_{\rm 2,eff}$  (s$^{-1}$)')

ScalingFactor = 1e-3 * 2 * np.pi * FI5.sfrq * FI5.gamma[FI5.nuc] / FI5.gamma['1H']  # rad/sec -> ppb

axs[1].errorbar(FI5.data['CS'][:,0]*1e6, FI5.data['CS'][:,1] / ScalingFactor, yerr=FI5.data['CS'][:,2] / ScalingFactor, c='r', marker='o', linestyle='' )
axs[1].plot(FI5.data['CS'][:,0]*1e6, Omegaeff / ScalingFactor, 'r--')
#axs[1].set_xlabel(r' [NS5A]  ($\mu$M)')
axs[1].set_ylabel(r' $\delta_{\rm eff}$  (ppb)')

axs[2].errorbar(FI5.data['R1'][:,0]*1e6, FI5.data['R1'][:,1], yerr=FI5.data['R1'][:,2], c='r', marker='o', linestyle='' )
axs[2].plot(FI5.data['R1'][:,0]*1e6, R1calc, 'r--')
axs[2].set_xlabel(r' [NS5A]  ($\mu$M)')
axs[2].set_ylabel(r' $R_{1}$  (s$^{-1}$)')

plt.savefig('exp_vs_fit.png',  dpi=300)
plt.clf()
plt.close()