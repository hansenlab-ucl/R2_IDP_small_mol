import os, sys
import nmrglue as ng
import numpy as np
import glob
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from lmfit import Parameters, minimize, report_fit, Model
from numpy import exp, linspace, random


class fit_raw: 
    # read in all bruker data from a directory using nmrglue and process it
    def read_data(self, directory):
        dic = {}
        data = {}
        file_list = glob.glob(directory + '/*.ft2')
        for f in file_list:
            name = f.split('/')[-1].split('.')[0]
            dic_key = 'd_' + name
            data_key = 'd_' + name
            dic[dic_key], data[data_key] = ng.pipe.read(f)
            # discard imaginary part
            data[data_key] = ng.proc_base.di(data[data_key])
            # reverse the data
            data[data_key] = ng.proc_base.rev(data[data_key])

        return dic, data

    # define a lorentzian function for a given frequency (f), center frequency (f0), linedwidth (w), phase (p), amplitude (A), and linear baseline correction slope (m) and intercept (b)
    def lorz_lincorr(self, x=None, f0=None, w=None, p=None, A=None, m=None, b=None):
        return A/(np.pi*w)*(np.cos(p)*(1/(1+((2*np.pi*(f0-x))/(np.pi*w))**2))-np.sin(p)*(((2*np.pi*(f0-x))/(np.pi*w)))/(1+((2*np.pi*(f0-x))/(np.pi*w))**2))+m*x+b

    def fit_DOSY_slices(self, dic=None, data_list=None, offset=-125):
        # Extract the frequency and intensity data from the first file in the list
        udic = ng.pipe.guess_udic(dic, data_list)
        ppm_real = ng.fileiobase.unit_conversion(udic[1]['size'],True,udic[1]['sw'],udic[1]['obs'],udic[1]['car']).ppm_scale() 

        def model_dataset(params, i, x):
            f0 = params['f0_%i' % (i+1)]
            w = params['w_%i' % (i+1)]
            p = params['p_%i' % (i+1)]
            A = params['A_%i' % (i+1)]
            m = params['m_%i' % (i+1)]
            b = params['b_%i' % (i+1)]
            return self.lorz_lincorr(x, f0, w, p, A, m, b)

        def objective(params, x, data):
            ndata, _ = data.shape
            resid = 0.0*data[:]

            # make residual per data set
            for i in range(ndata):
                resid[i, :] = data[i, :] - model_dataset(params, i, x)

            # now flatten this to a 1D array, as minimize() needs
            return resid.flatten()


        data = data_list[:]

        fit_params = Parameters()
        for iy, y in enumerate(data):
            fit_params.add('f0_%i' % (iy+1), value=4.75, min=4.66, max=4.82)
            fit_params.add('w_%i' % (iy+1), value=0.04)
            fit_params.add('p_%i' % (iy+1), value=0)
            fit_params.add('A_%i' % (iy+1), value=data_list[iy].max())
            fit_params.add('m_%i' % (iy+1), value=0)
            fit_params.add('b_%i' % (iy+1), value=0)
        # share parameters between slices
        for iy in range(2,data_list.shape[0]+1):
            fit_params['f0_%i' % iy].expr = 'f0_1'
            fit_params['w_%i' % iy].expr = 'w_1'
            fit_params['p_%i' % iy].expr = 'p_1'
    
        out = minimize(objective, fit_params, args=(ppm_real, data))
        #report_fit(out.params) #uncomment for reporting params

        plt.figure()
        for i in range(data_list.shape[0]):
            y_fit = model_dataset(out.params, i, ppm_real)
            plt.plot(ppm_real, data[i, :], 'o', ppm_real, y_fit, '-')
            plt.xlim(4.825, 4.65)
        plt.show()

        A_vals = []
        for i in range(1,data_list.shape[0]+1):
            A_vals.append(out.params.valuesdict()['A_{}'.format(i)])

        return np.asarray(A_vals)

    # define exponential function to fit inversion recovery intensities 
    def dosy(self, x=None, Diff=1e-09): # Diff=1e-09 is in m2/s, x=gradient values should be in Tm-1
        gamma = 267.522*1000000 #1H gyromag ratio s-1T-1
        big_del = 0.2 #diffusion time in s
        tau = 0.0002 # time to phase/rephase bipolar gradients s
        little_del = 0.002 #gradient pulse length s
        return exp(-Diff*gamma**2*x**2*little_del**2*(big_del-little_del/3-tau/2))

    def fit_DOSY_data(self, grads=np.asarray([0.681,1.998,3.315,4.632,5.949,7.265,8.582,9.899,11.216,12.533,13.850,15.166,16.483,17.800,19.117,20.434]), ydat=None):
        
        mod = Model(self.dosy, independent_vars=['x'])
        out = mod.fit(ydat, x=grads/100, Diff=1e-09) # grads/100 converts from Gcm-1 to Tm-1
        fit = out.best_fit
        D = out.params.valuesdict()['Diff'] 
        D_err = out.params['Diff'].stderr
        print(out.fit_report())
        plt.plot(grads/100, ydat, 'bo', label='data') # grads/100 converts from Gcm-1 to Tm-1
        plt.plot(grads/100, out.best_fit, 'r-', label='fit') # grads/100 converts from Gcm-1 to Tm-1
        plt.xlabel("grad (T/m)")
        plt.ylabel("I/I_0")
        plt.show()


        return fit, D, D_err


