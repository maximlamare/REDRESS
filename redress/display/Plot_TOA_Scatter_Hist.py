#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 08:01:03 2019

@author: rus
"""
from osgeo import gdal
from pathlib import Path
#import cartopy.crs as ccrs
#from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy import stats
from plot_funcs import (open_sat_model, getdata, transpt, save_geotiff,
                        collocate_rasters, rmse)


def plot_fig3(satprod, modelprod, snowmask, band, pos, basedir,date, typesimu,
              scatlim, histlim, title, cutoff=0.0):
    # Get bands of interest
    sat = satprod[band]
    ct_full = modelprod[band]
    # Get extent of outline in image coords
    rmin = pos['Start_large'][0]
    rmax = pos['End_large'][0]
    cmin = pos['Start_large'][1]
    cmax = pos['End_large'][1]
    # print (sat)
    # print (ct_full)
    # print("==============")
    # Prepare scatter
    satb= sat#[rmin:rmax, cmin:cmax]
    ctb = ct_full#[rmin:rmax, cmin:cmax]
    # print (satb)
    # print (ctb)
    print(snowmask)
    snowmask = snowmask['isSnow']#[rmin:rmax,cmin:cmax]
    print(rmin,rmax, cmin,cmax)
    masked_satb = np.ma.masked_where(snowmask <= cutoff, satb, np.nan)
    masked_ctb = np.ma.masked_where(snowmask <= cutoff, ctb, np.nan)
    print("Sat")
    print(np.min(masked_satb), np.max(masked_satb))
    print("Model")
    print(np.min(masked_ctb), np.max(masked_ctb))
    # Bias
    bias = masked_ctb - masked_satb

    # Std
    mean_bias = np.nanmean(bias)
    std_bias = np.nanstd(bias)
    low = mean_bias - 2*std_bias
    high = mean_bias + 2*std_bias

    # Stats        
    rmseval = rmse(masked_satb, masked_ctb)    
    # slope, intercept, r_value,\
    #    p_value, std_err = stats.mstats.linregress(masked_satb, masked_ctb,
    #                                               )
    slope, intercept, r_value,\
       p_value, std_err = stats.mstats.linregress(masked_satb[(bias > low) & (bias < high)],
                                       masked_ctb[(bias > low) & (bias < high)])
 
  
    # Start plot
    pt = plt.figure(figsize=(8, 8))
    
    # Set gridspec
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1,1])
    
    # Set axes
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
#    ax3 = plt.subplot(gs[1, :], projection=map_crs)

    # Plot scatters
    # Scatter 510
    ax1.scatter(masked_satb[(bias > low) & (bias < high)],
                masked_ctb[(bias > low) & (bias < high)],
                color="black",
                s=16, marker='.', alpha=0.2)
    ax1.plot([0, 1000], [0, 1000], color="lightblue", linewidth=0.8,
             linestyle='-', alpha=1)
    # ax1.scatter(masked_satb[bias > high],
    #             masked_ctb[bias > high],
    #             color="indianred",
    #             s=16, marker='.', alpha=0.4)
    # ax1.scatter(masked_satb[bias < low],
    #             masked_ctb[bias < low],
    #              color="steelblue",
    #             s=16, marker='.', alpha=0.4)
    # ax1.scatter(masked_satb[snowmask < 0.5],
    #             masked_ctb[snowmask < 0.5],
    #               color="green",
    #             s=16, marker='.', alpha=0.4)
#    ax1.scatter(masked_satb[snowmask !=1],
#                masked_ctb[snowmask !=1],
#                  color="green",
#                s=16, marker='.', alpha=0.4)
    ax1.scatter(masked_satb,
                masked_ctb,
                  color="green",
                s=16, marker='.', alpha=0.4)


    # Plot histograms
    # Hist 510
    n, bins, patches = ax2.hist(bias.flatten(), bins=100)
    for c, p in zip(bins, patches):
        if c < low:
            plt.setp(p, 'facecolor', 'steelblue')
        elif c > high:
            plt.setp(p, 'facecolor', 'indianred')
        elif c > low and c < high:
            plt.setp(p, 'facecolor', 'grey')


    ax2.plot([0, 0], [0, 10000], color='lightblue', linestyle='-', linewidth=1)

    # Adjust axes AX1 and AX2, scatter and histogram 510 nm
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
    
        if ax == ax1:
            ax.set_xlim(scatlim[0], scatlim[1])
            ax.set_ylim(scatlim[0], scatlim[1])
            ax.yaxis.set_major_locator(MultipleLocator(100))
            ax.yaxis.set_minor_locator(MultipleLocator(50))
            ax.set_xlabel(r"satellite TOA radiance,"
                          " mWm$^{-2}$sr$^{-1}$nm$^{-1}$",
                          fontsize=10)
            ax.set_ylabel(r"simulated TOA radiance, mWm$^{-2}$sr$^{-1}$nm$^{-1}$",
                          fontsize=10)
            ax.text(0.05, 0.92, "a)", fontsize=10, transform=ax.transAxes)
            ax.text(0.95, 0.02, "r$^{2}$ = %s"
                    "\n"
                    "RMSE = %s\n"
                    r"y = %sx + %s" % (round(r_value ** 2, 3), round(rmseval, 3),
                                       round(slope, 3), round(intercept, 3),
                                       ),
                    fontsize=10, transform=ax.transAxes, ha='right')
            ax.text(0.05, 0.85, "N = %s" % (masked_satb.count()), fontsize=10,
                    transform=ax.transAxes)
            r=" "
            if (typesimu == "MaskSnow"):
                r="(r = 0.2)"
            ax.set_title("%s : %s in %s %s "%(date,band,typesimu,r), fontdict={"fontsize": 10})
            ax.plot(-1,-1, '.', color="steelblue",
                    markersize=4, marker='.', alpha=0.4, label=r"Error < 2$\sigma$ of the bias")
            ax.plot(-1,-1, '.', color="indianred",
                    markersize=4, marker='.', alpha=0.4, label=r"Error > 2$\sigma$ of the bias")
             

        else:
            ax.set_xlim(histlim[0], histlim[1])
            ax.set_ylim(0, histlim[2])
            ax.yaxis.set_major_locator(MultipleLocator(50))
            ax.yaxis.set_minor_locator(MultipleLocator(25))
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set_ylabel("pixel count",
                          fontsize=10)
            ax.set_xlabel(r"simulation - satellite, Wm$^{-2}$sr$^{-1}\mu$m$^{-1}$",
                          fontsize=10)
            ax.text(0.05, 0.92, "b)", fontsize=10, transform=ax.transAxes)
            ax.text(0.05, 0.85,"Bias = %s" % (round(mean_bias, 3)),
                    fontsize=10, transform=ax.transAxes)
    
        ax.tick_params(axis='both', which='both', direction='inout', top=True,
                       right=True, left=True)
        ax.tick_params(axis='both', labelsize=10)
       
    
    # Save fig
    pt.savefig(str(basedir.joinpath(date+"/"+typesimu+"Results_f3_%s.pdf" % band)),
               dpi=300, facecolor='w', edgecolor='w',
               orientation='landscape', format='pdf',
               transparent=False, bbox_inches='tight', pad_inches=0.1,
               metadata=None)
    

    
dates=["20180213", "20180221", "20180316", "20180321", "20180406","20180417"]
date = "20180213"
for date in dates:
    # Specify data dir
    basedir = Path("/home/nheilir/REDRESS/REDRESS_files/outputs/outputs_Case5")#"/reflectance02/")
    cases, sats = open_sat_model(basedir.joinpath(date))
    angles, ds_angles = getdata(basedir.joinpath("%s/angles.tiff" %date))
    topobands, ds_topo = getdata(basedir.joinpath("%s/topobands.tiff" %date))
    snowmask2, ds_snowmask = getdata(basedir.joinpath("%s/subpixelsnow.tiff"  %date))
    # snowmask2, ds_snowmask = getdata("/home/nheilir/REDRESS/REDRESS_files/Entree_modele/%s/subpixelsnow.tif" %date)
    # Convert to image coordinates
    # Set coords to delimitate areas
    coords = [("Start_large", 45.12019109, 6.18545664),
              ("End_large", 44.97596878, 6.45833216),
              ]
    
        
    # Convert to image coordinates
    pos = {}
    for crd in coords:
        pos.update({crd[0]: transpt(sats['geotrans'], crd)[1:3]})
    simucase = ["_","_"]
    # plot_fig3(sats['sat_bands'],cases['toa_level_FullSnow'],snowmask2,
    #               "Oa05", pos,  basedir, date, simucase[0], [0, 600], [-340, 340, 260], "Band 05: 510 nm (Full snow)")
    
    # plot_fig3(sats['sat_bands'], cases['toa_level_FullSnow'],snowmask2,
    #               "Oa21", pos,  basedir, date , simucase[0], [0, 160], [-160, 160, 200], "Band 21: 1020nm (Full snow)")
    
    plot_fig3(sats['sat_bands'],cases['toa_level'],snowmask2,
                  "Oa05", pos, basedir, date , simucase[1], [0, 600], [-340, 340, 260], "Band 05: 510 nm (Mask snow)")
    
    plot_fig3(sats['sat_bands'], cases['toa_level'],snowmask2,
                  "Oa21", pos,  basedir, date, simucase[1], [0, 160], [-160, 160, 200], "Band 21: 1020nm (Mask snow)")
    



