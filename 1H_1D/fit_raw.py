# Import useful stuff
import os, sys
import nmrglue as ng
import numpy as np
import glob
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots


class fit_raw: 
    # Read in all bruker data from a directory using nmrglue and process it
    def read_data(self, directory):
        dic = {}
        data = {}
        file_list = glob.glob(directory + '/*.ft2')
        for f in file_list:
            name = f.split('/')[-1].split('.')[0]
            dic_key = 'd_' + name
            data_key = 'd_' + name
            dic[dic_key], data[data_key] = ng.pipe.read(f)
            data[data_key] = ng.process.proc_bl.base(data[data_key], np.arange(6000,6500,50))
        return dic, data

    # Define a lorentzian function for a given frequency (f), center frequency (f0), linedwidth (w), phase (p), amplitude (A), and linear baseline correction slope (m) and intercept (b)
    def lorz_lincorr(self, x=None, f0=None, w=None, p=None, A=None, m=None, b=None):
        return A/(np.pi*w)*(np.cos(p)*(1/(1+((2*np.pi*(f0-x))/(np.pi*w))**2))-np.sin(p)*(((2*np.pi*(f0-x))/(np.pi*w)))/(1+((2*np.pi*(f0-x))/(np.pi*w))**2))+m*x+b

    # Define a function to fit 1D spectra to a Lorentzian curve
    def fit_1D(self, dic=None, data=None, offset=4.773):
        # Extract the frequency and intensity data from the file
        udic = ng.pipe.guess_udic(dic, data)
        ppm_real = ng.fileiobase.unit_conversion(udic[0]['size'],True,udic[0]['sw'],udic[0]['obs'],udic[0]['car']).ppm_scale()  
        intensity = data
        # Select the indices of the intensity array where ppm values fall within [6.46, 6.7]
        ppm_range = (ppm_real >= 6.46) & (ppm_real <= 6.7)
        intensity_range = intensity[ppm_range]

        # Make an initial guess for the fit parameters
        f0_guess = 6.56
        w_guess = 0.01 
        p_guess = 0 
        A_guess = intensity_range.max() 
        m_guess = 0
        b_guess = 0 
        initial_guess = [f0_guess, w_guess, p_guess, A_guess, m_guess, b_guess]

        # Set bounds for fit
        bounds = ([6.46, 0, -np.inf, 0, -np.inf, -np.inf], [6.7, 0.1, np.inf, 3*intensity_range.max(), np.inf, np.inf])


        # Fit the Lorentzian function to the data
        popt, pcov = curve_fit(self.lorz_lincorr, ppm_real[ppm_range], intensity[ppm_range], p0=initial_guess, bounds=bounds)
        perr = np.sqrt(np.diag(pcov))
        # Generate the fitted curve using the optimized parameters
        fit_curve = self.lorz_lincorr(ppm_real[ppm_range], *popt)

        # Plot the original data and the fitted curve
        fig, ax = subplots()
        ax.plot(ppm_real[ppm_range], intensity[ppm_range], 'b-', label='Original data')
        ax.plot(ppm_real[ppm_range], fit_curve, 'r-', label='Fitted curve')
        ax.legend()
        ax.set_xlabel('ppm')
        ax.set_ylabel('Intensity')
        ax.set_title('Fitting Results')
        ax.set_xlim(6.7, 6.46)
        plt.show()

        # Return the fitted parameters and the fitted curve
        return popt, perr, fit_curve
