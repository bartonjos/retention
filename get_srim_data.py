# -*- coding: utf-8 -*-
"""
Created on Thu May  3 15:48:18 2018

@author: bartonjo

Get the SRIM data out of the txt files and convert the data into 
useable units.

"""
import sys
s = '/Users/bartonjo/PyFiles/LP/'
if s not in sys.path:
    sys.path.insert(0, s)
from cookbook import savitzky_golay as smooth
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# path where all of the data produced by SRIM is located
path = '/Users/bartonjo/PyFiles/SRIM/'
folder = path + 'Si-on-W-12MeV-5000ions-kp-40dsp/'

# Experimental data
fluence = 5e18  # ions/m2

# get damage data from file
try:
    nv = pd.read_table(folder + 'NOVAC.txt', skiprows=22, 
                       delimiter='\s+')
    nv.columns = ['depth_A','repl/A/ion']
    varows=27
except:
    nv = {'repl/A/ion':0}
    varows=26
    pass
va = pd.read_table(folder + 'VACANCY.txt', engine='python', 
                   skiprows=varows, skipfooter=2, delimiter='\s+')
va.columns = ['depth_A', 'pko/A/ion', 'vac/A/ion']

# get density data from VACACY file
with open(folder + 'VACANCY.txt', 'r') as fin:   
    for line in fin.readlines():
        row = line.strip().split()
        if ('Density' in row):
            atden = float(row[5])*1e6  # convert to m-3

# convert data and put in one dataframe of depth vs dpa
x = va.depth_A*1e-10*1e6  # convert to microns
vac = va['pko/A/ion'] + va['vac/A/ion'] - nv['repl/A/ion']
vac = vac * 1e10  # convert to number/m/ion
dpa = vac*fluence/atden*2

sdpa = smooth(np.array(dpa),15,3)

dam = pd.DataFrame({'Depth_um':x, 'dpa':dpa, 'smooth_dpa':sdpa})
dam.to_csv(folder + 'dpa_data.csv',index=False)

# roughly plot the data
plt.figure()
plt.plot(x,dpa,'s',label='TRIM data')
plt.plot(x,sdpa,'--k')
plt.xlabel(r'Depth [$\mu$m]', fontsize=18)
plt.ylabel('dpa', fontsize=18)
plt.tick_params(labelsize=18)
plt.title(folder[len(path):-1])
plt.legend(loc='best', fontsize=18)
plt.subplots_adjust(bottom=.12, left=.14)
plt.show()
plt.savefig(folder+'dpa_img.svg', format='svg')



