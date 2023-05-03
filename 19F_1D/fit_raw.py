import os, sys
import nmrglue as ng
import numpy as np
import glob
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots


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
            #data[data_key] = ng.process.proc_autophase.autops(data[data_key], 'acme')
            # reverse the data
            data[data_key] = ng.proc_base.rev(data[data_key])

        return dic, data

    # define a lorentzian function for a given frequency (f), center frequency (f0), linedwidth (w), phase (p), amplitude (A), and linear baseline correction slope (m) and intercept (b)
    def lorz_lincorr(self, x=None, f0=None, w=None, p=None, A=None, m=None, b=None):
        return A/(np.pi*w)*(np.cos(p)*(1/(1+((2*np.pi*(f0-x))/(np.pi*w))**2))-np.sin(p)*(((2*np.pi*(f0-x))/(np.pi*w)))/(1+((2*np.pi*(f0-x))/(np.pi*w))**2))+m*x+b

    # define a function to fit 1D spectra to a Lorentzian curve
    def fit_1D(self, dic=None, data=None, offset=-120):
        # Extract the frequency and intensity data from the file
        udic = ng.pipe.guess_udic(dic, data)
        ppm_real = np.linspace(offset+-(udic[0]['sw']/udic[0]['obs'])/2, offset+(udic[0]['sw']/udic[0]['obs'])/2, num=data.shape[0])
        intensity = data

        # Select the indices of the intensity array where ppm values fall within [-127, -124.5]
        ppm_range = np.logical_and(ppm_real >= -127, ppm_real <= -124.5)
        intensity_range = intensity[ppm_range]

        # Make an initial guess for the fit parameters
        f0_guess = -125.5
        w_guess = 0.08 
        p_guess = 0 
        A_guess = intensity_range.max() 
        m_guess = 0
        b_guess = 0 
        initial_guess = [f0_guess, w_guess, p_guess, A_guess, m_guess, b_guess]

        # set bounds for fit
        bounds = ([-126.5, 0, -np.inf, 0, -np.inf, -np.inf], [-124.5, 0.09, np.inf, 3*intensity_range.max(), np.inf, np.inf])


        # Fit the Lorentzian function to the data
        popt, pcov = curve_fit(self.lorz_lincorr, ppm_real[ppm_range], intensity[ppm_range], p0=initial_guess, bounds=bounds)
        perr = np.sqrt(np.diag(pcov))
        # Generate the fitted curve using the optimized parameters
        fit_curve = self.lorz_lincorr(ppm_real, *popt)

        # Plot the original data and the fitted curve
        fig, ax = subplots()
        ax.plot(ppm_real, intensity, 'b-', label='Original data')
        ax.plot(ppm_real, fit_curve, 'r-', label='Fitted curve')
        ax.legend()
        ax.set_xlabel('ppm')
        ax.set_ylabel('Intensity')
        ax.set_title('Fitting Results')
        ax.set_xlim(-124.5, -126.5)
        plt.show()

        # Return the fitted parameters and the fitted curve
        return popt, perr, fit_curve
