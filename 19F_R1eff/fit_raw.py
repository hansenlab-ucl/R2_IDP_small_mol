import os, sys
import nmrglue as ng
import numpy as np
import glob
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from lmfit import Parameters, minimize, report_fit


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
            # autophase correction
            data[data_key] = ng.process.proc_autophase.autops(data[data_key], 'acme')
            # reverse the data
            data[data_key] = ng.proc_base.rev(data[data_key])

        return dic, data

    # define a lorentzian function for a given frequency (f), center frequency (f0), linedwidth (w), phase (p), amplitude (A), and linear baseline correction slope (m) and intercept (b)
    def lorz_lincorr(self, x=None, f0=None, w=None, p=None, A=None, m=None, b=None):
        return A/(np.pi*w)*(np.cos(p)*(1/(1+((2*np.pi*(f0-x))/(np.pi*w))**2))-np.sin(p)*(((2*np.pi*(f0-x))/(np.pi*w)))/(1+((2*np.pi*(f0-x))/(np.pi*w))**2))+m*x+b

    def fit_IR_slices(self, dic=None, data_list=None, offset=-125):
        # Extract the frequency and intensity data from the first file in the list
        udic = ng.pipe.guess_udic(dic, data_list)
        ppm_real = np.linspace(offset+-(udic[1]['sw']/udic[1]['obs'])/2, offset+(udic[1]['sw']/udic[1]['obs'])/2, num=data_list[0].shape[0])

        # Select the indices of the intensity array where ppm values fall within [-127, -124.5]
        ppm_range = np.logical_and(ppm_real >= -127, ppm_real <= -124.5)

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


        data = data_list[:, ppm_range]

        fit_params = Parameters()
        for iy, y in enumerate(data):
            fit_params.add('f0_%i' % (iy+1), value=-125.5, min=-127, max=-124.5)
            fit_params.add('w_%i' % (iy+1), value=0.04)
            fit_params.add('p_%i' % (iy+1), value=0)
            fit_params.add('A_%i' % (iy+1), value=data_list[iy][ppm_range].min())
            fit_params.add('m_%i' % (iy+1), value=0)
            fit_params.add('b_%i' % (iy+1), value=0)
        # share parameters between slices
        for iy in range(2,data_list.shape[0]+1):
            fit_params['f0_%i' % iy].expr = 'f0_1'
            fit_params['w_%i' % iy].expr = 'w_1'
            fit_params['p_%i' % iy].expr = 'p_1'
    
        out = minimize(objective, fit_params, args=(ppm_real[ppm_range], data))
        #report_fit(out.params) #uncomment for reporting params

        plt.figure()
        for i in range(data_list.shape[0]):
            y_fit = model_dataset(out.params, i, ppm_real[ppm_range])
            plt.plot(ppm_real[ppm_range], data[i, :], 'o', ppm_real[ppm_range], y_fit, '-')
            plt.xlim(-125.2, -125.8)
        plt.show()

        A_vals = []
        for i in range(1,data_list.shape[0]+1):
            A_vals.append(out.params.valuesdict()['A_{}'.format(i)])

        return np.asarray(A_vals)

    # define exponential function to fit inversion recovery intensities 
    def r1(self, t=None, M1_t0=None, M1_eq=None, R1=None):
        return (M1_t0-M1_eq)*(np.exp(-R1*t))+M1_eq

    def fit_IR_data(self, t=np.asarray([.078125,.15625,.3125,.625,1.25,2.5,5,10,20]), y=None):
        p0 = (-20000, 20000, 0.2) #initial guess
        bounds = ([-np.inf, 0, 0], [0, np.inf, np.inf])
        popt, pcov = curve_fit(self.r1, t, y, p0=p0, bounds=bounds)
        R1 = popt[2]
        R1_error = np.sqrt(np.diag(pcov))[2]
        y_fit = self.r1(t, *popt)
        plt.plot(t, y, 'bo', label='data')
        plt.plot(t, y_fit, 'r-', label='fit')
        plt.xlabel('Time (s)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.show()
        return popt, pcov, R1, R1_error


