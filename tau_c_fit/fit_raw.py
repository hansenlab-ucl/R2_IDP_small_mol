"""
  Class object and functions to extract chemical shifts, R1_eff, R2_eff, and 
  DOSY observables from raw data in support of 'Small-molecule binding to 
  intrinsically disordered proteins revealed by experimental NMR 19F 
  transverse spin-relaxation' by Heller, Shukla, Figueiredo, and Hansen.
  
"""
# import useful tools
import os, sys
import nmrglue as ng
import numpy as np
import glob

class fit_raw:
  # read in all bruker data from a directory using nmrglue
  def read_data(directory):
      dic = {}
      data = {}
      file_list = glob.glob(directory + '/*.ft2')
      for f in file_list:
          name = 'd_' + f.split('/')[-1].split('.')[0]
          dic[name], data[name] = ng.pipe.read(f)
      return dic, data

    # define function to read and processes raw 1D spectra with a given phase correction
  def Proc1D(self, data=None, phase=55):
    # phase correction
    data = ng.proc_base.ps(data, p0=phase)
    # discard imaginary part
    data = ng.proc_base.di(data)
    # autophase correction
    data = ng.process.proc_autophase.autops(data, 'acme')
    # reverse the data
    data = ng.proc_base.rev(data)
    return data

    # define a lorentzian function for a given frequency (f), center frequency (f0), linedwidth (w), phase (p), and amplitude (A)
    def lorz(self, f=None, f0=None, w=None, p=None, A=None):
      return A/(np.pi*w)*(np.cos(p)*(1/(1+((2*np.pi*(f0-f))/(np.pi*w))**2))-np.sin(p)*(((2*np.pi*(f0-f))/(np.pi*w)))/(1+((2*np.pi*(f0-f))/(np.pi*w))**2))
