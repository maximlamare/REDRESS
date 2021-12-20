#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Rugged Terrain code.

This script run the rugged terrain model for testing and debugging purposes.
"""
import warnings
from pathlib import Path
import sys
import os
import time
import numpy as np
import netCDF4 as nc4
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import glob

sys.path.append("/home/nheilir/REDRESS/REDRESS")

from redress.inputs.satellites import import_s3OLCI
from redress.inputs.geotiffs import import_DEM_gtiff
from redress.geospatial.gdal_ops import (build_poly_from_geojson,
                                                resample_dataset,
                                                write_xarray_dset)
from redress.inputs.products import Model,SMD

import re
import redress.display.mask_res_plot as fplot
import redress.optics.generic_algorithm as ga
from redress.display.plot_funcs import (open_sat_model, getdata, transpt, rmse)


warnings.filterwarnings('ignore')
Sattype="sat2"
demfile="lautaretdem5mtopcenter.tif"
#demfile="lautaretdem1mtop.tif"

demname=os.path.splitext(demfile)[0]
outputs_folder = "/home/nheilir/REDRESS/REDRESS_files/outputs/outputs"+Sattype+demname+"/"
# Check whether the specified path exists or not

if not os.path.exists(outputs_folder): 
  # Create a new directory because it does not exist 
  os.makedirs(outputs_folder)
  print("The new directory is created!")
  
#outputs_folder = "/home/nheilir/REDRESS/REDRESS_files/outputs_iteration_FixBug_diffconst/"
inputs_folder =  "/home/nheilir/REDRESS/REDRESS_files/Entree_modele/"

# #  constant definition
#20180417 les valeurs sont unconues cas 1 0.477   1.381
#measure_value = {
#                 '20180213' :{'aod':0.02 , 'ozone':0.6 , 'water':0.48 , 'fixssa':41.41}, #casOK
##                '20180213' :{'aod':0.02 , 'ozone':0.3954 , 'water':0.175 , 'fixssa':41.41},#cas initial
#                 '20180221' :{'aod':0.03 , 'ozone':0.5 , 'water':1.5 , 'fixssa':33.66}, #casOK
##                 '20180221' :{'aod':0.15 , 'ozone':0.3867 , 'water':0.331 , 'fixssa':33.66},#cas initial
#                 '20180314' :{'aod':0.005 , 'ozone':0.3966 , 'water':0.9 , 'fixssa':45.42},#casOK
#                 '20180316' :{'aod':0.01 , 'ozone':0.6 , 'water':1.7 , 'fixssa':27.37},#copy22
##                 '20180314' :{'aod':0.02 , 'ozone':0.3966 , 'water':0.230 , 'fixssa':45.42},#cas initial
#                 '20180320' :{'aod':0.01 , 'ozone':0.6 , 'water':1.7 , 'fixssa':27.37},#copy22
#                 '20180321' :{'aod':0.01 , 'ozone':0.6 , 'water':1.7 , 'fixssa':27.37},#copy22
#                 '20180322' :{'aod':0.01 , 'ozone':0.6 , 'water':1.7 , 'fixssa':27.37},#casOK
##                 '20180322' :{'aod':0.05 , 'ozone':0.4126 , 'water':0.379 , 'fixssa':27.37},#cas initial
#                 '20180406' :{'aod':0.09 , 'ozone':0.347 , 'water':0.512 , 'fixssa':5.91}, #cas initial OK
#                 '20180417' :{'aod':0.15 , 'ozone':0.08 , 'water':3. , 'fixssa':5.91}, #casOK
##                 '20180417' :{'aod':0.09 , 'ozone':0.347 , 'water':0.512 , 'fixssa':5.91}, #cas initial
#                 }
measure_value = {
                 '20180420' :{'aod':0.15 , 'ozone':0.08 , 'water':3. , 'fixssa':5.91}, #casOK
                 }

#mask_bands=[4,5,6,7,13,14,15,18,19,20]
mask_bands=[1,2,3,4,5,6,7,8,9,10,11,12]

#wvl_nobands = [400.0, 412.5, 442.5, 
#               665.0,
#                       673.75, 681.25, 708.75, 753.75, 778.75, 865.0, 
#                       1020.0] a garder,
#               
#[400.0, 412.5, 442.5, 490.0, 510.0, 560.0, 620.0, 665.0, 673.75, 681.25, 708.75, 753.75, 761.25, 764.375, 767.5, 778.75,
# 865.0, 885.0, 900.0, 940.0, 1020.0]

osmd=SMD()
#osmd.init_DEM(outputs_folder,inputs_folder, "dem.tif","gdal_reader", "large.geojson" )
#Extent large: (6.187097, 44.966847) - (6.452219, 45.128622)


osmd.init_DEM(outputs_folder,inputs_folder, demfile,"gdal_reader" )

#osmd.init_DEM(outputs_folder,inputs_folder, "dem.tif", "large.geojson","gdal_reader" )
############################SAT3#######################################################
#si nous voulons simuler une seule image sat on choisis parmis cette liste et activer la boucle for i in range(1):
#sinon il va simuler tout les sat dans inputs_folder+"SAT/
# select sat date 
#Sattype="sat3"
#date = "20180213"
#s3_str =inputs_folder+"SAT/S3A_OL_1_EFR____20180213T102022_20180213T102322_20180214T135709_0180_028_008_2160_LN1_O_NT_002.SEN3/xfdumanifest.xml"

#date = "20180221"
#s3_str =inputs_folder+"SAT/S3A_OL_1_EFR____20180221T101253_20180221T101553_20180222T152018_0179_028_122_2160_LN1_O_NT_002.SEN3/xfdumanifest.xml"
###
#date = "20180406"
#s3_str =inputs_folder+"SAT/S3A_OL_1_EFR____20180406T093144_20180406T093444_20180407T142434_0180_029_364_2160_LN1_O_NT_002.SEN3/xfdumanifest.xml"

#date = "20180322" 
#s3_str =inputs_folder+"SAT/S3A_OL_1_EFR____20180322T092031_20180322T092331_20180323T145246_0180_029_150_2160_LN1_O_NT_002.SEN3/xfdumanifest.xml"
#        
#date="20180316"
#s3_str = inputs_folder+"SAT/S3A_OL_1_EFR____20180316T101637_20180316T101937_20180317T150123_0180_029_065_2160_LN1_O_NT_002.SEN3/xfdumanifest.xml"
#
#date="20180314"
#s3_str = inputs_folder+"SAT/S3A_OL_1_EFR____20180314T092800_20180314T093100_20180315T130347_0179_029_036_2160_LN1_O_NT_002.SEN3/xfdumanifest.xml"
##
#date="20180320"
#s3_str = inputs_folder+"SAT/S3A_OL_1_EFR____20180320T101253_20180320T101553_20180321T140307_0179_029_122_2160_LN1_O_NT_002.SEN3/xfdumanifest.xml"
##
#date="20180321"
#s3_str = inputs_folder+"SAT/S3A_OL_1_EFR____20180321T094642_20180321T094942_20180322T132053_0179_029_136_2160_LN1_O_NT_002.SEN3/xfdumanifest.xml"
##
#
#date="20180417"
#s3_str = inputs_folder+"SAT/S3A_OL_1_EFR____20180417T094642_20180417T094942_20180418T134454_0179_030_136_2160_LN1_O_NT_002.SEN3/xfdumanifest.xml"

#SENTINEL 2 
date="20180420"
s3_str = inputs_folder+"SAT2/L1C/%s"%date

############################SAT2#######################################################

# select sat date  or all date in the folders
pattern='EFR____(.+?)T'
for i in range(1):
#for s3_str in glob.iglob(inputs_folder+"SAT/*.SEN3/xfdumanifest.xml"):
#    date = re.search(pattern, s3_str).group(1)

    if not os.path.exists(outputs_folder+date):
        os.makedirs(outputs_folder+date) 
#    if (date=="20180314" or date=="20180322"):
#        osmd.init_SAT(outputs_folder,inputs_folder, date, s3_str,"esa_snap")
#    else:
    osmd.init_SAT(outputs_folder,inputs_folder, date, s3_str,"pytroll_satpy",Sattype,demname)

#    selected_band_name,selected_band=[],[]
#    for index in [4,5,6,10,11,12]:
#        selected_band_name.append(osmd.sat.meta["bandlist"][index])
#        selected_band.append(osmd.sat.meta["wavelengths"][index])
#    osmd.sat.meta["bandlist"]=selected_band_name
#    osmd.sat.meta["wavelengths"]=selected_band
    # Set up Model
    osmd.model=Model()    
    osmd.model.ssa=np.full(osmd.sat.shape,measure_value[osmd.sat.date]['fixssa'])
    osmd.model.angles = osmd.sat.angles        
    osmd.model.topo_bands = osmd.sat.topo_bands        
    osmd.model.meta = osmd.sat.meta
    osmd.model.date = osmd.sat.date        
    
    osmd.model.import_rt_model("python_6S")        
    # Calculate albedo
    osmd.model.set_rt_options("Urban", aod=measure_value[osmd.sat.date]['aod'],
                              refl=0,                           
                              ozone=measure_value[osmd.sat.date]['ozone'],
                              water= measure_value[osmd.sat.date]['water']
                              )

    osmd.sat.sat_band_np=np.zeros((len(osmd.model.meta["wavelengths"]),osmd.sat.shape[0],osmd.sat.shape[1]))
    temprefl=np.zeros((len(osmd.model.meta["wavelengths"]),osmd.sat.shape[0],osmd.sat.shape[1]))
    if (Sattype=="sat2"):
        def calcul_dt(date):
            import jdcal
            dt=1./np.power((1-0.01673*np.cos(0.0172*(sum(jdcal.gcal2jd(date[0:4], date[4:6], date[6:9]))-2))), 2)
            return dt
        dt=calcul_dt(date)
#         rho_kE_sd(t)*cos(theta_s)/pi
        for band,index in zip(osmd.model.meta["bandlist"],range(len(osmd.model.meta["wavelengths"]))):
            temprefl[index,:,:]= osmd.sat.bands[band].values
            osmd.sat.bands[band].values    = ((osmd.sat.bands[band].values/100.)*osmd.sat.meta["solar_irradiance"][band]*dt*np.cos(osmd.sat.angles["SZA"]).values)/np.pi   
            osmd.sat.sat_band_np[index,:,:]= osmd.sat.bands[band].values            
   
    elif Sattype=="sat3":
        for band,index in zip(osmd.model.meta["bandlist"],range(len(osmd.model.meta["wavelengths"]))):
            osmd.sat.sat_band_np[index,:,:]= osmd.sat.bands[band].values
    
    for iterOptim in range(1,11):
        print("iteration", iterOptim)
        osmd.model.simulate_TOA_radiance(osmd.model) 

        ga.obs(osmd) #calculate rtild
        ga.refl_estimate(osmd) #calculate r
        ga.ssa_estimate(osmd, None,model="2-param") #calculate direct_model
        ga.mask_estimate(osmd)
        # Save final dataset
        write_xarray_dset(osmd.model.snowmask, outputs_folder+"%s/subpixelsnow%s.tiff"% (osmd.sat.date,iterOptim), 
                          2154, osmd.sat.geotransform)
        write_xarray_dset(osmd.model.toa_rad, outputs_folder+"%s/toa_level%s.tiff" %(osmd.sat.date,iterOptim), 
                          2154, osmd.sat.geotransform)
        fplot.ProjectSnowMaskSAT2(inputs_folder,'20180420',outputs_folder, osmd,iterOptim)  
        fplot.PlotSsaMaskSnow(outputs_folder, osmd, iterOptim)
#        Plots in iteration 

        fplot.PlotSsaMaskSnow(outputs_folder, osmd, iterOptim)
        fplot.PlotHistGenerateMask(outputs_folder, osmd, iterOptim)
    
#    y=10
#    x=20        
#    fplot.PlotDMPix(outputs_folder,"Neige",osmd,iterOptim,x,y,reflectance=temprefl)
#    fplot.PlotDMPixRADIANCE(outputs_folder,"Neige",osmd,iterOptim,x,y)
#
#    y=120
#    x=12      #non neige
#    fplot.PlotDMPix(outputs_folder,"ombréNONneige",osmd,iterOptim,x,y,reflectance=temprefl)
#    fplot.PlotDMPixRADIANCE(outputs_folder,"ombréNONneige",osmd,iterOptim,x,y)
##    
#    y=120
#    x=22    #non neige
#    fplot.PlotDMPix(outputs_folder,"ombréneige",osmd,iterOptim,x,y,reflectance=temprefl)
#    fplot.PlotDMPixRADIANCE(outputs_folder,"ombréneige",osmd,iterOptim,x,y)
    
    


   
    
#    fplot.CompareSnowMask(inputs_folder,'20180420',outputs_folder, osmd, True)
    
    
        ax=xr.Dataset()  
        
        ax["a"]=xr.DataArray(osmd.model.a,dims=['y', 'x'],coords=osmd.model.topo_bands.coords)
        
        write_xarray_dset(ax, outputs_folder+"%s/a.tiff" %(osmd.sat.date), 
                      2154, osmd.sat.geotransform)
        
        rmsex=xr.Dataset()  
        
        rmsex["rmse"]=xr.DataArray(osmd.model.rmse,dims=['y', 'x'],coords=osmd.model.topo_bands.coords)
        
        write_xarray_dset(rmsex, outputs_folder+"%s/rmse.tiff" %(osmd.sat.date), 
                      2154, osmd.sat.geotransform)
    
    
        
#        #############Compare with SAT2
#        if date=='20180213' :
#            dateSAT2="20180214"
#            fplot.CompareSnowMask(inputs_folder,dateSAT2,outputs_folder, osmd, True)
#            fplot.ProjectSnowMaskSAT2(inputs_folder,dateSAT2,outputs_folder, osmd)
#            dateSAT2="20180219"
#            fplot.CompareSnowMask(inputs_folder,dateSAT2,outputs_folder, osmd, True)
#            fplot.ProjectSnowMaskSAT2(inputs_folder,dateSAT2,outputs_folder, osmd)
#        elif date=='20180221':
#            dateSAT2="20180219"
#            fplot.CompareSnowMask(inputs_folder,dateSAT2,outputs_folder, osmd, True)
#            fplot.ProjectSnowMaskSAT2(inputs_folder,dateSAT2,outputs_folder, osmd)
#            dateSAT2="20180224"
#            fplot.CompareSnowMask(inputs_folder,dateSAT2,outputs_folder, osmd, True)
#            fplot.ProjectSnowMaskSAT2(inputs_folder,dateSAT2,outputs_folder, osmd)
#        elif date=='20180314':
#            dateSAT2="20180316"
#            fplot.CompareSnowMask(inputs_folder,dateSAT2,outputs_folder, osmd, True)
#            fplot.ProjectSnowMaskSAT2(inputs_folder,dateSAT2,outputs_folder, osmd)
#        elif date=='20180316':
#            dateSAT2="20180316"
#            fplot.CompareSnowMask(inputs_folder,dateSAT2,outputs_folder, osmd, True)
#            fplot.ProjectSnowMaskSAT2(inputs_folder,dateSAT2,outputs_folder, osmd)
#        elif date=='20180320':
#            dateSAT2="20180316"
#            fplot.CompareSnowMask(inputs_folder,dateSAT2,outputs_folder, osmd, True)
#            fplot.ProjectSnowMaskSAT2(inputs_folder,dateSAT2,outputs_folder, osmd)
#            dateSAT2="20180321"
#            fplot.CompareSnowMask(inputs_folder,dateSAT2,outputs_folder, osmd, True)
#            fplot.ProjectSnowMaskSAT2(inputs_folder,dateSAT2,outputs_folder, osmd)
#        elif date=='20180321':
#            dateSAT2="20180321"
#            fplot.CompareSnowMask(inputs_folder,dateSAT2,outputs_folder, osmd, True)
#            fplot.ProjectSnowMaskSAT2(inputs_folder,dateSAT2,outputs_folder, osmd)
#        elif date=='20180322':
#            dateSAT2="20180321"
#            fplot.CompareSnowMask(inputs_folder,dateSAT2,outputs_folder, osmd, True)
#            fplot.ProjectSnowMaskSAT2(inputs_folder,dateSAT2,outputs_folder, osmd)
#            dateSAT2="20180326"
#            fplot.CompareSnowMask(inputs_folder,dateSAT2,outputs_folder, osmd, True)
#            fplot.ProjectSnowMaskSAT2(inputs_folder,dateSAT2,outputs_folder, osmd)
#        elif date=='20180406':
#            dateSAT2="20180405"
#            fplot.CompareSnowMask(inputs_folder,dateSAT2,outputs_folder, osmd, True)
#            fplot.ProjectSnowMaskSAT2(inputs_folder,dateSAT2,outputs_folder, osmd)
#            dateSAT2="20180410"
#            fplot.CompareSnowMask(inputs_folder,dateSAT2,outputs_folder, osmd, True)
#            fplot.ProjectSnowMaskSAT2(inputs_folder,dateSAT2,outputs_folder, osmd)
#        elif date=='20180417':
#            dateSAT2="20180415"
#            fplot.CompareSnowMask(inputs_folder,dateSAT2,outputs_folder, osmd, True)
#            fplot.ProjectSnowMaskSAT2(inputs_folder,dateSAT2,outputs_folder, osmd)
#            dateSAT2="20180420"
#            fplot.CompareSnowMask(inputs_folder,dateSAT2,outputs_folder, osmd, True)
#            fplot.ProjectSnowMaskSAT2(inputs_folder,dateSAT2,outputs_folder, osmd)
#    
    
    #    y=28
    #    x=49
    #    fplot.PlotDMPix(outputs_folder,"ombré",osmd,iterOptim,x,y,mask_bands)
    #    fplot.PlotDMPixRADIANCE(outputs_folder,"ombré",osmd,iterOptim,x,y)
    #        
    
                    
    #        y=5
    #        x=66        
    #        fplot.PlotDMPix(outputs_folder,"Plat",osmd,iterOptim,x,y)#,mask_bands)
    #        fplot.PlotDMPixRADIANCE(outputs_folder,"Plat",osmd,iterOptim,x,y)
    #        
    #        y=39
    #        x=31
    #        fplot.PlotDMPix(outputs_folder,"Plat",osmd,iterOptim,x,y)#,mask_bands)
    #        fplot.PlotDMPixRADIANCE(outputs_folder,"Plat",osmd,iterOptim,x,y)






