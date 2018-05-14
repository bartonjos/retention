# -*- coding: utf-8 -*-
"""
Created on Tue May  8 14:43:46 2018

@author: bartonjo

Plot NRA data taken from Excel spreadsheet so that plots don't look
like they were plotted with an Excel spreadsheet.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = '/Users/bartonjo/Documents/Logs/UFG W exp 2017/' + \
        'retention exp/nra_data/'

df = pd.read_csv(path + 'nra_from_wampler.csv')

srimpath = '/Users/bartonjo/PyFiles/SRIM/' + \
            'Si-on-W-12MeV-5000ions-kp-40dsp/dpa_data.csv'
srim = pd.read_csv(srimpath)

# relevant dictionaries

buttons = {0:'I-7', 1:'I-3', 2:'I-4', 
           3:'U-5', 4:'U-1', 5:'U-2', 6:'U-4'}
           
props = {'I-7':'ITER W, 0.6 dpa, inner', 
         'I-3':'ITER W, 0.6 dpa, middle',
         'I-4':'ITER W, 0 dpa, outer', 'U-5':'UFG W, 0.6 dpa, inner', 
         'U-1':'UFG W, 0 dpa, middle', 'U-2':'UFG W, > 0.6 dpa, outer', 
         'U-4':'UFG W, 0.06 dpa, middle'}

cols = {'I-7':df.columns[7], 'I-3':df.columns[6], 'I-4':df.columns[5], 
           'U-5':df.columns[2], 'U-1':df.columns[1], 
           'U-2':df.columns[3], 'U-4':df.columns[4]}

sym = {0:'o', 1:'^', 2:'D', 3:'s', 4:'x', 5:'*', 6:'>'}
hue = {0:'b', 1:'g', 2:'r', 3:'c', 4:'m', 5:'y', 6:'k'}

# plot the nra and srim data
plt.figure()
fsize = 22  # axis font size

plt.subplot(2,1,1)
for i in [2,1,0]:
    plt.plot(df[df.columns[0]], np.array(df[cols[buttons[i]]])*1e3, 
             '-'+sym[i]+hue[i], markersize=10,
             label=props[buttons[i]])
plt.plot(srim.Depth_um, np.array(srim.smooth_dpa)*1e3/1.5e2, '--k',
         label='0.6 dpa/150')
#plt.yscale('log')
plt.ylim([0,5])
plt.xlim([-.3,3.5])
plt.ylabel(r'D/W $\times 10^{-3}$', fontsize=fsize)
plt.tick_params(labelsize=fsize-2)
plt.legend(bbox_to_anchor=(1.27,1.))


plt.subplot(2,1,2)
for i in [4,3,5,6]:
    plt.plot(df[df.columns[0]], np.array(df[cols[buttons[i]]])*1e3, 
             '-'+sym[i]+hue[i], markersize=10,
             label=props[buttons[i]])
plt.plot(srim.Depth_um, np.array(srim.smooth_dpa)*1e3/1.5e2, '--k')
#plt.yscale('log')
plt.ylim([0,5])
plt.xlim([-.3,3.5])
plt.ylabel(r'D/W $\times 10^{-3}$', fontsize=fsize)
plt.xlabel(r'Depth [$\mu$m]', fontsize=fsize)
plt.tick_params(labelsize=fsize-2)
plt.legend(bbox_to_anchor=(1.27,1.))
plt.subplots_adjust(bottom=.13, left=.1, right=.81, top=.97)














