# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:28:17 2017

@author: bartonjo

Get RGA generated data files into a dataframe and plot the raw signals.
Then use a function to save the 'blessed' data.
"""
import sys
s = '/Users/bartonjo/PyFiles/LP'
if s not in sys.path:
    sys.path.insert(0, s)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cookbook import savitzky_golay as smooth

# user defined strings
#filename = 'I-3_p6dpa_suspect'
#filename = 'I-4_0dpa_suspect'
#filename = 'I-7_p6dpa'
#filename = 'U-1_0dpa'
#filename = 'U-5_p6dpa'
#filename = 'U-2_p6dpa'
filename = 'U-4_p06dpa'
#filename = 'runtest_cem_on'
cem = 0  # 0=off  1=on  (=0 only for U-4_p06dpa)
path = '/Users/bartonjo/Documents/Logs/'
path = path + 'UFG W exp 2017/retention exp/tds_data/'

# define strings that are easier to use than the default headings
time = 'Elapsed Time'
temp = 'Temp.'
h2 = '  2'
hd = '  3'
d2 = '  4'
m7 = '  7'
h2o = ' 18'
co = ' 28'
m32 = ' 32'
m40 = ' 40'
co2 = ' 44'

# read in the data
df = pd.read_table(path + filename, header=3)

## snapshot of the raw data by plotting h molecules
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax2 = ax1.twiny()
#    # plot data vs time
#ax1.semilogy(df[time], df[h2], label=r'H$_2$')
#ax1.semilogy(df[time], df[hd], label='HD')
#ax1.semilogy(df[time], df[d2], label=r'D$_2$')
#ax1.set_xlim([0,3000])
#ax1.set_xlabel('time [s]')
#ax1.set_ylim([1e-12,1e-6])
#ax1.set_ylabel('partial pressure')
#    # add temperature axis
#ticks = np.array([100, 200, 400, 600, 800, 1000])
#tick_loc = [df[time][np.min(np.where(df[temp]>=i))] for i in ticks]
#tick_vals = ['%i' % z for z in ticks]
#ax2.set_xlim(ax1.get_xlim())
#ax2.set_xticks(tick_loc)
#ax2.set_xticklabels(tick_vals)
#ax2.set_xlabel('temperature [C]')
#ax1.legend(loc='best')
#plt.show()

# convert raw data to calibrated data with background subtraction
leak = 2.4e-11  # D2 mol/s  calibration leak
avo = 6.022e23  # at/mol
area = np.pi*(3e-3)**2
if cem==0:
    pp = 5.3e-9  # calibration pressure cem off
        # get background pressures from run with empty tube
    back = pd.read_table(path + 'RunTest3', header=3)
else:
    pp = 3.0e-9  # calibration pressure cem on
    back = pd.read_table(path + 'runtest_cem_on', header=3)
    # get calibration factor
cal = leak*avo/pp/area
    # smooth the background pressure noise level
h2back = smooth(np.array(back[h2]),51,3)
hdback = smooth(np.array(back[hd]),51,3)
d2back = smooth(np.array(back[d2]),51,3)
    # make arrays the same length
diff = np.int(np.size(df[h2]) - np.size(h2back))
if diff > 0:  # data array is bigger than background array
    a = np.zeros(diff) + h2back[-1]
    h2back = np.append(h2back,a)
    a = np.zeros(diff) + hdback[-1]
    hdback = np.append(hdback,a)
    a = np.zeros(diff) + d2back[-1]
    d2back = np.append(d2back,a)
elif diff < 0:  # data array is smaller than background array
    h2back = h2back[:diff]
    hdback = hdback[:diff]
    d2back = d2back[:diff]
df[h2] = (df[h2] - h2back)*cal
df[hd] = (df[hd] - hdback)*cal
df[d2] = (df[d2] - d2back)*cal

# function that saves the "blessed" data
def tdssave(filename,key,df):
    data = pd.DataFrame({'time_s':df['Elapsed Time'], 
                         'temp_C':df['Temp.'],
                         'h2_m-2s-1':df['  2'], 
                         'hd_m-2s-1':df['  3'], 
                         'd2_m-2s-1':df['  4']})
    data.to_hdf(filename + '.h5',key)
    return

# function that integrates the data from tstart to tend
def tdsint(tstart=200, tend=2500, floor=1e14, t=df[time], 
           fd2=df[d2], fhd=df[hd]):
    fd2 = [fd2[i] if fd2[i]>floor else 0 for i in range(np.size(fd2))]
    fhd = [fhd[i] if fhd[i]>floor else 0 for i in range(np.size(fhd))]
    fd2 = np.array(fd2)
    fhd = np.array(fhd)
    f = 2*fd2 + .5*fhd
    zmin = np.min(np.where(t >= tstart))
    zmax = np.min(np.where(t >= tend))
    dt = np.int(np.mean(np.diff(t[zmin:zmax+1])))
    print(dt)
    return np.sum(f[zmin:zmax+1])*dt

# snapshot of the data by plotting h molecules
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()
    # plot data vs time
#ax1.semilogy(df[time], df[h2], 'b', label=r'H$_2$')
ax1.semilogy(df[time], df[hd], 'g', label='HD')
ax1.semilogy(df[time], df[d2], 'r', label=r'D$_2$')
ax1.semilogy([200,200],[1e14,1e19],'k--')
ax1.semilogy([2500,2500],[1e14,1e19],'k--')
ax1.set_xlim([0,3000])
ax1.set_xlabel('time [s]')
ax1.set_ylim([1e14,1e19])
ax1.set_ylabel(r'flux [m$^{-2}$ s$^{-1}$]')
    # add temperature axis
ticks = np.array([100, 200, 400, 600, 800, 1000])
tick_loc = [df[time][np.min(np.where(df[temp]>=i))] for i in ticks]
tick_vals = ['%i' % z for z in ticks]
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(tick_loc)
ax2.set_xticklabels(tick_vals)
ax2.set_xlabel('temperature [C]')
    # add title and fluence to plot
#plt.title(filename + '\n\n')
flu = '{:0.2e}'.format(tdsint())
unit = r' D atoms/m$^2$'
plt.text(1000,2.5e18,'Fluence = ' + flu + unit)
plt.text(1000,5e18, filename)
#plt.subplots_adjust(top=.88)
ax1.legend(loc='upper right')
plt.show()
#plt.savefig(filename)


# Save the data
#tdssave(filename,filename,df)





















