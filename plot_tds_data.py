# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 13:41:36 2018

@author: bartonjo

Scratch module to plot (and re-plot) TDS data that has already been 
calibrated and saved to hdf5 files.  Comments should tell the user 
what data was used in the plot.
It is asssumed this script is run in this directory:
/Users/bartonjo/Documents/Logs/UFG W exp 2017/retention exp/tds_data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Functions used by some plotting scripts------------------------------
def tdsint(x, t, tstart=200, tend=2500):
    '''
    Integrate a calibrated signal x over a time range tstart-tend.
    x and t is assumed to be a dataframe column
    '''
    x = np.array([x[i] if x[i]>0 else 0 for i in range(np.size(x))])
    t = np.array(t)
    zmin = np.min(np.where(t >= tstart))
    zmax = np.min(np.where(t >= tend))
    dt = np.int(np.mean(np.diff(t[zmin:zmax+1])))
#    print('dt = ' + str(dt) + ' s')
    return np.sum(x[zmin:zmax+1])*dt

def combineD(d2, hd, floor=1e14):
    '''
    Combine the calibrated D2 and HD signals in order to get the 
    total D flux.
    d2 is the dataframe column having the D2 flux data
    hd is the dataframe column having the HD flux data
    '''
    d2 = [d2[i] if d2[i]>floor else 0 for i in range(np.size(d2))]
    hd = [hd[i] if hd[i]>floor else 0 for i in range(np.size(hd))]
    d2 = np.array(d2)
    hd = np.array(hd)
    return 2*d2 + .5*hd

def int_regions(x, temp, time, temp1=50, tmid=500, temp2=1000):
    '''
    Integrate regions of the spectra given a starting, middle, and 
    ending temperature for a given signal, x.
    '''
    z1 = np.min(np.where(temp >= temp1))
    zmid = np.min(np.where(temp >= tmid))
    z2 = np.min(np.where(temp >= temp2))
    tot = tdsint(x, time, tstart=time[z1], tend=time[z2])
    mid = tdsint(x, time, tstart=time[z1], tend=time[zmid])
    frac = mid/tot  # fraction of x out at temperature = tmid 
    return tot, mid, frac

# Plotting scripts-----------------------------------------------------

# tds file dictionary
buttons = {0:'I-7', 1:'I-3', 2:'I-4', 
           3:'U-5', 4:'U-1', 5:'U-2', 6:'U-4'}

tdsfile = {'I-7':'I-7_p6dpa.h5', 'I-4':'I-3_p6dpa_suspect.h5',
         'I-3':'I-4_0dpa_suspect.h5', 'U-5':'U-5_p6dpa.h5', 
         'U-1':'U-1_0dpa.h5', 'U-2':'U-2_p6dpa.h5', 
         'U-4':'U-4_p06dpa.h5'}

props = {'I-7':'ITER W, 0.6 dpa, inner', 
         'I-3':'ITER W, 0.6 dpa, middle',
         'I-4':'ITER W, 0 dpa, outer', 'U-5':'UFG W, 0.6 dpa, inner', 
         'U-1':'UFG W, 0 dpa, middle', 'U-2':'UFG W, > 0.6 dpa, outer', 
         'U-4':'UFG W, 0.06 dpa, middle'}

sym = {0:'o', 1:'^', 2:'D', 3:'s', 4:'x', 5:'*', 6:'>'}
hue = {0:'b', 1:'g', 2:'r', 3:'c', 4:'m', 5:'y', 6:'k'}

## ---------------------------------------------------------------------
## Comparison of UFG and ITER W retention with 0.6 dpa damage
## ---------------------------------------------------------------------
#b1, b2, b3 = 3, 1, 0  #4, 2, 5 #3, 1, 0
#
#ufgd = pd.read_hdf(tdsfile[buttons[b1]])
#iterd = pd.read_hdf(tdsfile[buttons[b2]])
#iterdi = pd.read_hdf(tdsfile[buttons[b3]])
#zmaxu = np.min(np.where(ufgd.temp_C == np.max(ufgd.temp_C)))
#zmaxi = np.min(np.where(iterd.temp_C == np.max(iterd.temp_C)))
#zmaxii = np.min(np.where(iterdi.temp_C == np.max(iterdi.temp_C)))
#
## integrate regions of the spectra
#tmid=500
#ufgw = combineD(ufgd['d2_m-2s-1'],ufgd['hd_m-2s-1'])
#utot, umid, ufrac = int_regions(ufgw,
#                                ufgd.temp_C, ufgd.time_s, tmid=tmid)
#iterw = combineD(iterd['d2_m-2s-1'],iterd['hd_m-2s-1'])
#itot, imid, ifrac = int_regions(iterw,
#                                iterd.temp_C, iterd.time_s, tmid=tmid)
#iterwi = combineD(iterdi['d2_m-2s-1'],iterdi['hd_m-2s-1'])
#itoti, imidi, ifraci = int_regions(iterwi,
#                                iterdi.temp_C, iterdi.time_s,tmid=tmid)
#                                
## print retention numbers
#print(utot)
#print(itot)
#print(itoti)
#
## plot data
#plt.figure()
#fsize = 22  # axis font size
#plt.plot(ufgd.temp_C[:zmaxu], ufgw[:zmaxu]/1e17, '-'+hue[b1])
#plt.plot(ufgd.temp_C[:zmaxu:10], ufgw[:zmaxu:10]/1e17, 
#         '-'+hue[b1]+sym[b1], markersize=8,
#         label=props[buttons[b1]])
#plt.plot(iterd.temp_C[:zmaxi], iterw[:zmaxi]/1e17, '-'+hue[b2])
#plt.plot(iterd.temp_C[:zmaxi:10], iterw[:zmaxi:10]/1e17, 
#         '-'+hue[b2]+sym[b2], markersize=9,
#         label=props[buttons[b2]])
#plt.plot(iterdi.temp_C[:zmaxi], iterwi[:zmaxi]/1e17, '-'+hue[b3])
#plt.plot(iterdi.temp_C[:zmaxi:10], iterwi[:zmaxi:10]/1e17, 
#         '-'+hue[b3]+sym[b3], markersize=8,
#         label=props[buttons[b3]] + ' (High T exposure)')
#plt.plot([tmid,tmid],[-.5,12],'--k')
#plt.xlim([0,1020])
#plt.ylim([-.5,4.5])
#s = 'D inventory left after T = %i C' % tmid
#s = s + '\nUFG-W: %1.2f' % ((utot-umid)/1e20)
#s = s + r'$\times$ 10$^{20}$ D/m$^2$'
#s = s + '\nITER-W: %1.2f' % ((itot-imid)/1e20)
#s = s + r'$\times$ 10$^{20}$ D/m$^2$'
#s = s + '\nITER-W (high T): %1.2f' % ((itoti-imidi)/1e20)
#s = s + r'$\times$ 10$^{20}$ D/m$^2$'
#plt.text(tmid+10,-0.45,s)
## plot formatting
#plt.tick_params(labelsize=fsize-2)
#plt.xlabel(r'Temperature [C]', fontsize=fsize)
#plt.ylabel(r'D flux [$10^{17}$ m$^{-2}$s$^{-1}$]', fontsize=fsize)
#plt.legend(bbox_to_anchor=(.8,1.14))
##plt.title('Tungsten samples pre-damaged with a\n 12 MeV Si ion beam' +\
##            ' to 0.6 dpa')
#plt.subplots_adjust(bottom=0.12)
#plt.text(940,4.1,'(c)', fontsize=fsize)
#plt.show()


## ---------------------------------------------------------------------
## Comparison of UFG and ITER W retention with no damage
## ---------------------------------------------------------------------
#
## button index:
#ib = 2
#ub = 4
#
#ufgd = pd.read_hdf(tdsfile[buttons[ub]])
#iterd = pd.read_hdf(tdsfile[buttons[ib]])
#zmaxu = np.min(np.where(ufgd.temp_C == np.max(ufgd.temp_C)))
#zmaxi = np.min(np.where(iterd.temp_C == np.max(iterd.temp_C)))
## integrate regions of the spectra
#tmid=500
#ufgw = combineD(ufgd['d2_m-2s-1'],ufgd['hd_m-2s-1'])
#utot, umid, ufrac = int_regions(ufgw,
#                                ufgd.temp_C, ufgd.time_s, tmid=tmid)
#iterw = combineD(iterd['d2_m-2s-1'],iterd['hd_m-2s-1'])
#itot, imid, ifrac = int_regions(iterw,
#                                iterd.temp_C, iterd.time_s, tmid=tmid)
## plot data
#plt.figure()
#plt.plot(ufgd.temp_C[:zmaxu], ufgw[:zmaxu]/1e17,
#         label=props[buttons[ub]])
#plt.plot(iterd.temp_C[:zmaxi], iterw[:zmaxi]/1e17,
#         label=props[buttons[ib]])
#
#plt.plot([tmid,tmid],[-.5,12],'--k')
#s = 'D inventory left after T = %i C' % tmid
#s = s + '\nUFG-W: %1.2f' % ((utot-umid)/1e20)
#s = s + r'$\times$ 10$^{20}$ D/m$^2$'
#s = s + '\nITER-W: %1.2f' % ((itot-imid)/1e20)
#s = s + r'$\times$ 10$^{20}$ D/m$^2$'
#plt.text(tmid+10,3,s)
#
## plot formatting
#plt.xlim([0,1020])
#plt.ylim([-.5,6])
#plt.tick_params(axis='x', labelsize=18)
#plt.tick_params(axis='y', labelsize=18)
#plt.xlabel(r'Temperature [$^o$C]', fontsize=18)
#plt.ylabel(r'D flux [$10^{17}$ m$^{-2}$s$^{-1}$]', fontsize=18)
#plt.legend(loc='best')
##plt.title('Tungsten samples pre-damaged with a\n 12 MeV Si ion beam' +\
##            ' to 0.6 dpa')
#plt.subplots_adjust(bottom=0.12)
#plt.show()




## ---------------------------------------------------------------------
## plots of retention from ITER W buttons
## ---------------------------------------------------------------------
#
## get the data
#iteri = pd.read_hdf(tdsfile[buttons[0]])
#iterm = pd.read_hdf(tdsfile[buttons[1]])
#itero = pd.read_hdf(tdsfile[buttons[2]])
#zmaxi = np.min(np.where(iteri.temp_C == np.max(iteri.temp_C)))
#zmaxm = np.min(np.where(iterm.temp_C == np.max(iterm.temp_C)))
#zmaxo = np.min(np.where(itero.temp_C == np.max(itero.temp_C)))
#
## integrate regions of the spectra
#tmid=500
#inn = combineD(iteri['d2_m-2s-1'],iteri['hd_m-2s-1'])
#toti, midi, fraci = int_regions(inn,
#                                iteri.temp_C, iteri.time_s, tmid=tmid)
#mid = combineD(iterm['d2_m-2s-1'],iterm['hd_m-2s-1'])
#totm, midm, fracm = int_regions(mid,
#                                iterm.temp_C, iterm.time_s, tmid=tmid)
#out = combineD(itero['d2_m-2s-1'],itero['hd_m-2s-1'])
#toto, mido, fraco = int_regions(out,
#                                itero.temp_C, itero.time_s, tmid=tmid)
#
## plot data
#plt.figure()
#fsize = 22  # axis font size
#plt.plot(iteri.temp_C[:zmaxi], inn[:zmaxi]/1e17, '-'+hue[0])
#plt.plot(iteri.temp_C[:zmaxi:10], inn[:zmaxi:10]/1e17,
#         '-'+hue[0]+sym[0], markersize=8,
#         label=props[buttons[0]] + ' (High T exposure)')
#plt.plot(iterm.temp_C[:zmaxm], mid[:zmaxm]/1e17, '-'+hue[1])
#plt.plot(iterm.temp_C[:zmaxm:10], mid[:zmaxm:10]/1e17,
#         '-'+hue[1]+sym[1], markersize=9,
#         label=props[buttons[1]])
#plt.plot(itero.temp_C[:zmaxo], out[:zmaxo]/1e17, '-'+hue[2])
#plt.plot(itero.temp_C[:zmaxo:10], out[:zmaxo:10]/1e17, 
#         '-'+hue[2]+sym[2], markersize=8,
#         label=props[buttons[2]] + ' (High T exposure)')
#
#
##plt.plot([tmid,tmid],[-.5,12],'--k')
##s = 'D inventory left after T = %i C' % tmid
##s = s + '\n' + 'in' + ': %1.2f' % ((toti-midi)/1e20)
##s = s + r'$\times$ 10$^{20}$ D/m$^2$'
##s = s + '\nmid: %1.2f' % ((totm-midm)/1e20)
##s = s + r'$\times$ 10$^{20}$ D/m$^2$'
##s = s + '\nout: %1.2f' % ((toto-mido)/1e20)
##s = s + r'$\times$ 10$^{20}$ D/m$^2$'
##plt.text(tmid+10,6,s)
#
## plot formatting
#plt.xlim([0,1020])
#plt.ylim([-.5,4.5])
#plt.tick_params(labelsize=fsize-2)
#plt.xlabel('Temperature [C]', fontsize=fsize)
#plt.ylabel(r'D flux [$10^{17}$ m$^{-2}$s$^{-1}$]', fontsize=fsize)
#plt.legend(bbox_to_anchor=(.8,1.14))
##plt.title('Tungsten samples ...')
#plt.subplots_adjust(bottom=0.12)
#plt.text(940,4.1,'(a)', fontsize=fsize)
#plt.show()


# ---------------------------------------------------------------------
# plots of retention from UFG W buttons
# ---------------------------------------------------------------------

# get the data
ufgi = pd.read_hdf(tdsfile[buttons[3]])
ufgm = pd.read_hdf(tdsfile[buttons[4]])
ufgo = pd.read_hdf(tdsfile[buttons[5]])
ufgm2 = pd.read_hdf(tdsfile[buttons[6]])
zmaxi = np.min(np.where(ufgi.temp_C == np.max(ufgi.temp_C)))
zmaxm = np.min(np.where(ufgm.temp_C == np.max(ufgm.temp_C)))
zmaxo = np.min(np.where(ufgo.temp_C == np.max(ufgo.temp_C)))
zmaxm2 = np.min(np.where(ufgm2.temp_C == np.max(ufgm2.temp_C)))

# integrate regions of the spectra
tmid=500
inn = combineD(ufgi['d2_m-2s-1'],ufgi['hd_m-2s-1'])
toti, midi, fraci = int_regions(inn,
                                ufgi.temp_C, ufgi.time_s, tmid=tmid)
mid = combineD(ufgm['d2_m-2s-1'],ufgm['hd_m-2s-1'])
totm, midm, fracm = int_regions(mid,
                                ufgm.temp_C, ufgm.time_s, tmid=tmid)
out = combineD(ufgo['d2_m-2s-1'],ufgo['hd_m-2s-1'])
toto, mido, fraco = int_regions(out,
                                ufgo.temp_C, ufgo.time_s, tmid=tmid)

mid2 = combineD(ufgm2['d2_m-2s-1'],ufgm2['hd_m-2s-1'])
totm2, midm2, fracm2 = int_regions(mid2,
                                ufgm2.temp_C, ufgm2.time_s, tmid=tmid)
                                
# plot data
plt.figure()
fsize = 22  # axis font size
plt.plot(ufgi.temp_C[:zmaxi], inn[:zmaxi]/1e17, '-'+hue[3])
plt.plot(ufgi.temp_C[:zmaxi:10], inn[:zmaxi:10]/1e17, 
         '-'+hue[3]+sym[3], markersize=8,
         label=props[buttons[3]])
plt.plot(ufgm.temp_C[:zmaxm], mid[:zmaxm]/1e17, '-'+hue[4])
plt.plot(ufgm.temp_C[:zmaxm:10], mid[:zmaxm:10]/1e17, 
         '-'+hue[4]+sym[4], markersize=9,
         label=props[buttons[4]])
plt.plot(ufgo.temp_C[:zmaxo], out[:zmaxo]/1e17, '-'+hue[5])
plt.plot(ufgo.temp_C[:zmaxo:10], out[:zmaxo:10]/1e17, 
         '-'+hue[5]+sym[5], markersize=9,
         label=props[buttons[5]])
         
plt.plot(ufgm2.temp_C[:zmaxm2], mid2[:zmaxm2]/1e17, '-'+hue[6])
plt.plot(ufgm2.temp_C[:zmaxm2:10], mid2[:zmaxm2:10]/1e17, 
         '-'+hue[6]+sym[6], markersize=9,
         label=props[buttons[6]])

#plt.plot([tmid,tmid],[-.5,12],'--k')
#s = 'D inventory left after T = %i C' % tmid
#s = s + '\n' + 'in' + ': %1.2f' % ((toti-midi)/1e20)
#s = s + r'$\times$ 10$^{20}$ D/m$^2$'
#s = s + '\nmid: %1.2f' % ((totm-midm)/1e20)
#s = s + r'$\times$ 10$^{20}$ D/m$^2$'
#s = s + '\nout: %1.2f' % ((toto-mido)/1e20)
#s = s + r'$\times$ 10$^{20}$ D/m$^2$'
#plt.text(tmid+10,6,s)

# plot formatting
plt.xlim([0,1020])
plt.ylim([-.5,11])
plt.tick_params(labelsize=fsize-2)
plt.xlabel('Temperature [C]', fontsize=fsize)
plt.ylabel(r'D flux [$10^{17}$ m$^{-2}$s$^{-1}$]', fontsize=fsize)
plt.legend(bbox_to_anchor=(.8,.9))  #loc='right')
#plt.title('Tungsten samples ...')
plt.subplots_adjust(bottom=0.12)
plt.text(940,10.1,'(b)', fontsize=fsize)
plt.show()











