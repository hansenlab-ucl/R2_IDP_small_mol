# Small-molecule Binding to an Intrinsically Disordered Protein Revealed by Experimental NMR 19F Transverse Spin-relaxation

This repository contains code, analysis scripts, a pulse sequence, and input files to recreate all figures for the manuscript entitled "Small-molecule Binding to an Intrinsically Disordered Protein Revealed by Experimental NMR 19F Transverse Spin-relaxation" by Heller, Shukla, Figueiredo, and Hansen.

## This repository contains: 
#### Pulse sequence  
1. Bruker pulse sequence for the pseudo-2D ligand-detected transverse (spin-spin, R2,eff) relaxation experiment in pulse_sequence/

#### Jupyter Notebooks  
2. Jupyter Notebook and code to create plots of 1D ligand-detected chemical shift data (19F and 1H NMR) in 1H_1D/ and 19F_1D/
3. Jupyter Notebook and code to create plots of pseudo-2D ligand-detected longitudinal (spin-lattice, R1,eff) relaxation (19F NMR) in 19F_R1eff/
4. Jupyter Notebook and code to create plots of pseudo-2D ligand-detected transverse (spin-spin, R2,eff) relaxation (19F NMR) in 19F_R2eff/
5. Jupyter Notebook and code to create plots of pseudo-2D Diffusion Ordered SpectroscopY (DOSY) (1H NMR) in 1H_DOSY/
6. Jupyter Notebook to create plots of CD data in CD/

#### Analysis Scripts
7. Scripts for the simultaneous analysis of 19F chemical shifts, DOSY, R1,eff, and R2,eff data in the context of rotational correlation times (tau_c) in tau_c_fit/

## How to use this code

Experimental data, including nuclear magnetic resonance (NMR) spectroscopy data files (.ft2 format) and circular dichroism (CD) data files (.txt format), should first be downloaded from Zenodo: [INSERT LINK HERE] and placed in a directory called Data in the same directory as this README.md file.

Jupyter Notebook directories 2-6 (above) are independent and self-contained. Executing all cells in 19F_1D/ 19F_R1eff/ 19F_R2eff/ will output dictionaries containing data to various json files in json/. Executing json/write_file.ipynb will combine these data into an input data file for the tau_c_fit here: tau_c_fit/data.inp.

## Dependencies
  * [Python=3.6.8](https://www.python.org/downloads/)
  * [NumPy=1.19.5](https://numpy.org/install/)
  * [nmrglue=0.9.dev0](https://nmrglue.readthedocs.io/en/latest/install.html)
  * [matplotlib=3.3.4](https://matplotlib.org/stable/users/installing/index.html)  
  * [lmfit=1.0.1](https://lmfit.github.io/lmfit-py/installation.html)  
  * [scipy=1.3.0](https://scipy.org/install/)  
  * [pathos=0.2.3](https://pypi.org/project/pathos/)  

The script has been written and tested with the above dependencies. Performance with other module versions has not been tested.


