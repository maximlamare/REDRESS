#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 06:44:48 2019

@author: rus
"""
from pathlib import Path
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from plot_funcs import (open_sat_model, getdata, transpt, rmse)
from datetime import datetime as dt
import pandas as pd

def plot_fig(all_data, band1, band2, outputfile):
    """Plot time series histograms."""
    # Start plot
    pt = plt.figure(figsize=(10, 8))

    # Set gridspec
    gs = gridspec.GridSpec(2, 6, wspace=0.0, hspace=0.17)

    # Set axes
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])
    ax4 = plt.subplot(gs[0, 3])
    ax5 = plt.subplot(gs[0, 4])
    ax5b = plt.subplot(gs[0, 5])
    
    ax6 = plt.subplot(gs[1, 0])
    ax7 = plt.subplot(gs[1, 1])
    ax8 = plt.subplot(gs[1, 2])
    ax9 = plt.subplot(gs[1, 3])
    ax10 = plt.subplot(gs[1, 4])
    ax10b = plt.subplot(gs[1, 5])
    
    # Plot histograms
    for ax, date in zip([ax1, ax2, ax3, ax4, ax5, ax5b], all_data.keys()):
        bias_full = all_data[date]["results"][band1]["bias_full"].flatten()
        # bias = bias[~bias.mask]
        bias_mask = all_data[date]["results"][band1]["bias_mask"].flatten()
        # bias_slope = bias_slope[~bias_slope.mask]

        ax.hist(bias_mask,
                bins=50, color="silver", alpha=1, 
                 weights=np.zeros_like(bias_mask)+1./ bias_mask.size,
                 label="mask model")
        ax.hist(bias_full,
                color="indianred",
                weights=np.zeros_like(bias_full)+1./ bias_full.size, bins=50, alpha=0.6,
                label="Full model")
        ax.text(0.95, 0.75, 
                    "rmse_full = %s\n"
                    "rmse_mask = %s"
                     % (round(all_data[date]["results"][band1]["rmse_full"], 3),
                     round(all_data[date]["results"][band1]["rmse_mask"], 3)),
                    fontsize=7, transform=ax.transAxes, ha='right',)
        ax.plot([0,0], [0, 1000], color="black", linestyle="--", linewidth=0.8)
        ax.set_xlim(-299, 299)
        ax.set_ylim(0, 0.16)
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.yaxis.set_major_locator(MultipleLocator(0.02))
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))
        ax.tick_params(axis='both', which='both', direction='inout', top=True,
                       right=False, left=True, bottom=True, labelsize=8)
        ax.set_title(dt.strftime(dt.strptime(date, "%Y%m%d"), "%d/%m/%Y"),
                     fontsize=12)
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            
    for ax, date in zip([ax6, ax7, ax8, ax9, ax10, ax10b], all_data.keys()):
        bias_full = all_data[date]["results"][band2]["bias_full"].flatten()
        # bias = bias[~bias.mask]
        bias_mask = all_data[date]["results"][band2]["bias_mask"].flatten()
        # bias_slope = bias_slope[~bias_slope.mask]

        ax.hist(bias_mask,
                bins=50, color="silver", alpha=1, 
                  weights=np.zeros_like(bias_mask)+1./ bias_mask.size)
        ax.hist(bias_full,
                color="indianred",
                weights=np.zeros_like(bias_full)+1./ bias_full.size, bins=50, alpha=0.6)
        ax.text(0.95, 0.75, 
                    "rmse_full = %s\n"
                    "rmse_mask = %s"
                     % (round(all_data[date]["results"][band2]["rmse_full"], 3),
                     round(all_data[date]["results"][band2]["rmse_mask"], 3)),
                    fontsize=7, transform=ax.transAxes, ha='right',)
        ax.plot([0,0], [0, 1000], color="black", linestyle="--", linewidth=0.8)
        ax.set_xlim(-145, 145)
        ax.set_ylim(0, 0.16)
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(25))
        ax.yaxis.set_major_locator(MultipleLocator(0.02))
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))
        ax.tick_params(axis='both', which='both', direction='inout', top=True,
                        right=True, left=True, bottom=True, labelsize=8)
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
        
        

    for ax in (ax2, ax3, ax4, ax5,ax5b, ax7, ax8, ax9, ax10, ax10b):
        ax.set_yticklabels([])
        
    for ax in (ax1, ax6):
        ax.set_ylabel("frequency", fontsize=10)
    for ax in (ax5b, ax10b):
        ax.tick_params(axis='both', which='both', direction='inout', top=True,
                       right=True, left=True, bottom=True, labelsize=8)
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
    ax5b.legend(loc=1, prop={'size': 7})
    ax8.set_title(r"simulation - satellite, mWm$^{-2}$sr$^{-1}$nm$^{-1}$", x=0.5, y=-0.25,
                  fontdict={"fontsize": 10})
    ax1.text(-0.5, 0.37, "510 nm", rotation='vertical',
                  fontdict={"fontsize": 12}, transform=ax1.transAxes)
    ax6.text(-0.5, 0.37, "1020 nm", rotation='vertical',
                  fontdict={"fontsize": 12}, transform=ax6.transAxes)

    pt.savefig(str(outputfile), dpi=300, facecolor='w', edgecolor='w',
               orientation='landscape', format='pdf',
               transparent=False, bbox_inches='tight', pad_inches=0.1,
               metadata=None)


# Specify data dir
basedir = Path("/home/nheilir/REDRESS/REDRESS_files/outputs_iteration/")

# Set coords to delimitate areas
coords = [("Start_large", 45.12019109, 6.18545664),
          ("End_large", 44.97596878, 6.45833216),
          ]

# Make def
all_data = {}
for sat_date in ["20180213", "20180221", "20180314", "20180322", "20180406", "20180417"]:
    # Convert to image coordinates
    pos = {}

    all_data[sat_date] = {"model": open_sat_model(basedir.joinpath(sat_date))[0],
                          "sat": open_sat_model(basedir.joinpath(sat_date))[1],
                          "angles": getdata(basedir.joinpath(sat_date, "angles.tiff"))[0],
                          "snowmask": getdata(basedir.joinpath(sat_date, "subpixelsnow.tiff"))[0]}
    for crd in coords:
        pos.update({crd[0]: transpt( all_data[sat_date]['sat']['geotrans'], crd)[1:3]})
    all_data[sat_date].update({"pos": pos})

    # Calculate for all bands
    all_data[sat_date].update({"results": {}})
    for band in all_data[sat_date]["sat"]["sat_bands"].keys():
        sat_band = all_data[sat_date]["sat"]["sat_bands"][band]
        ct_full = all_data[sat_date]["model"]["toa_level_FullSnow"][band]
        ct_mask = all_data[sat_date]["model"]["toa_level_MaskSnow"][band]
        # ct_flat_band = all_data[sat_date]["model"]["toa_level_1"][band]

        # # Get extent of outline in image coords
        # rmin = all_data[sat_date]['pos']['Start_large'][0]
        # rmax = all_data[sat_date]['pos']['End_large'][0]
        # cmin = all_data[sat_date]['pos']['Start_large'][1]
        # cmax = all_data[sat_date]['pos']['End_large'][1]

        # Prepare scatter
        satb = sat_band#[rmin:rmax, cmin:cmax]
        ctb_full = ct_full#[rmin:rmax, cmin:cmax]
        ctb_mask = ct_mask#[rmin:rmax, cmin:cmax]
        # ctb_flat = ct_flat_band[rmin:rmax, cmin:cmax]

        # Snowmask
        cutoff = 0.0
        snowmask = all_data[sat_date]["snowmask"]['altitude']#[rmin:rmax, cmin:cmax]
        masked_satb = np.ma.masked_where(snowmask <= cutoff, satb, np.nan)
        masked_ctb_full = np.ma.masked_where(snowmask <= cutoff, ctb_full, np.nan)
        masked_ctb_mask = np.ma.masked_where(snowmask <= cutoff, ctb_mask, np.nan)
        # masked_ctb_flat = np.ma.masked_where(snowmask <= cutoff, ctb_flat, np.nan)

        # Bias
        bias_full = masked_ctb_full - masked_satb
        bias_mask = masked_ctb_mask- masked_satb
        # bias_flat = masked_ctb_flat - masked_satb

        # Stats
        rmse_val_full = rmse(masked_satb, masked_ctb_full)
        rmse_val_mask = rmse(masked_satb, masked_ctb_mask)
        # rmse_val_flat = rmse(masked_satb, masked_ctb_flat)

        std_val_full = np.nanstd(bias_full)
        std_val_mask = np.nanstd(bias_mask)
        stat_val_full = stats.mstats.linregress(masked_satb, masked_ctb_full)
        stat_val_mask = stats.mstats.linregress(masked_satb, masked_ctb_mask)
        # Update dict
        all_data[sat_date]["results"].update({band: {"bias_full": bias_full,
                                                      "rmse_full": rmse_val_full,
                                                      "stats_full": stat_val_full,
                                                      "std_full": std_val_full,
                                                      "std_mask": std_val_mask,
                                                      "bias_mask": bias_mask,
                                                      "rmse_mask": rmse_val_mask,
                                                      "stats_mask": stat_val_mask}})
plot_fig(all_data, "Oa05", "Oa21", basedir.joinpath("Results_HistAllDate.pdf"))
     
biasses = pd.DataFrame(columns=all_data.keys(), index=all_data[list(all_data.keys())[0]]["results"].keys())
for date in all_data:
    print(date)
    for wvl in all_data[date]["results"]:
          print(wvl)
          biasses.loc[wvl, date] = all_data[date]["results"][wvl]["std_full"]
          print("bias mask %s = %s" % (wvl, np.nanmean(all_data[date]["results"][wvl]["rmse_mask"])))
          print("bias full %s = %s" % (wvl, np.nanmean(all_data[date]["results"][wvl]["rmse_full"])))

          biasses.to_csv(basedir.joinpath("std_SLP.csv"), sep=',')
