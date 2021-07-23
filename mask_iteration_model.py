#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Rugged Terrain code.

This script run the rugged terrain model for testing and debugging purposes.
"""
import warnings
from pathlib import Path
import sys
import os
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


warnings.filterwarnings('ignore')

outputs_folder = "/home/nheilir/REDRESS/REDRESS_files/outputs_iteration_FixBug_diffconst/"
inputs_folder =  "/home/nheilir/REDRESS/REDRESS_files/Entree_modele/"


# #  constant definition
#20180417 les valeurs sont unconues cas 1 0.477   1.381
measure_value = {
                 '20180213' :{'aod':0.02 , 'ozone':0.6 , 'water':0.48 , 'fixssa':41.41}, #cas5
#                '20180213' :{'aod':0.02 , 'ozone':0.6 , 'water':0.55 , 'fixssa':41.41}, #cas4
#                '20180213' :{'aod':0.02 , 'ozone':0.5 , 'water':0.5 , 'fixssa':41.41}, #cas3
#                '20180213' :{'aod':0.02 , 'ozone':0.480 , 'water':0.175 , 'fixssa':41.41}, #cas2
#                 '20180213' :{'aod':0.02 , 'ozone':0.477 , 'water':1.381 , 'fixssa':41.41},cas1 
#                 '20180213' :{'aod':0.02 , 'ozone':0.3954 , 'water':0.175 , 'fixssa':41.41},#cas initial
                 '20180221' :{'aod':0.15 , 'ozone':0.3867 , 'water':0.331 , 'fixssa':33.66},
                 '20180314' :{'aod':0.02 , 'ozone':0.3966 , 'water':0.230 , 'fixssa':45.42},
                 '20180322' :{'aod':0.05 , 'ozone':0.4126 , 'water':0.379 , 'fixssa':27.37},
                 '20180406' :{'aod':0.09 , 'ozone':0.347 , 'water':0.512 , 'fixssa':5.91}, 
                 '20180417' :{'aod':0.09 , 'ozone':0.347 , 'water':0.512 , 'fixssa':5.91}, }

mask_bands=[15,16,17,18,19,20,21]
case=5

osmd=SMD()

osmd.init_DEM(outputs_folder,inputs_folder, "dem.tif", "large.geojson","gdal_reader" )
#si nous voulons simuler une seule image sat on choisis parmis cette liste et activer la boucle for i in range(1):
#sinon il va simuler tout les sat dans inputs_folder+"SAT/
# select sat date 
date = "20180213"
s3_str =inputs_folder+"SAT/S3A_OL_1_EFR____20180213T102022_20180213T102322_20180214T135709_0180_028_008_2160_LN1_O_NT_002.SEN3/xfdumanifest.xml"

#date = "20180221"
#s3_str =inputs_folder+"SAT/S3A_OL_1_EFR____20180221T101253_20180221T101553_20180222T152018_0179_028_122_2160_LN1_O_NT_002.SEN3/xfdumanifest.xml"
#
#date = "20180406"
#s3_str =inputs_folder+"SAT/S3A_OL_1_EFR____20180406T093144_20180406T093444_20180407T142434_0180_029_364_2160_LN1_O_NT_002.SEN3/xfdumanifest.xml"
#
#date = "20180322"
#s3_str =inputs_folder+"SAT/S3A_OL_1_EFR____20180322T092031_20180322T092331_20180323T145246_0180_029_150_2160_LN1_O_NT_002.SEN3/xfdumanifest.xml"
#
#date="20180314"
#s3_str = inputs_folder+"SAT/S3A_OL_1_EFR____20180314T092800_20180314T093100_20180315T130347_0179_029_036_2160_LN1_O_NT_002.SEN3/xfdumanifest.xml"
#
#
#date="20180417"
#s3_str = inputs_folder+"SAT/S3A_OL_1_EFR____20180417T094642_20180417T094942_20180418T134454_0179_030_136_2160_LN1_O_NT_002.SEN3/xfdumanifest.xml"



# select sat date  or all date in the folders
pattern='EFR____(.+?)T'
for i in range(1):
#for s3_str in glob.iglob(inputs_folder+"SAT/*.SEN3/xfdumanifest.xml"):
#    date = re.search(pattern, s3_str).group(1)
    if not os.path.exists(outputs_folder+date):
        os.makedirs(outputs_folder+date) 
    if (date=="20180314" or date=="20180322"):
        osmd.init_SAT(outputs_folder,inputs_folder, date, s3_str,"esa_snap")
    else:
        osmd.init_SAT(outputs_folder,inputs_folder, date, s3_str,"pytroll_satpy")

   
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
                              refl=0, # AVOIR pourquoi c'est 0                              
                              ozone=measure_value[osmd.sat.date]['ozone'],
                              water= measure_value[osmd.sat.date]['water']
                              )
    
    
    for iterOptim in range(1,2):
        osmd.model.simulate_TOA_radiance(case,osmd) 
        
        osmd.sat.sat_band_np=np.zeros((len(osmd.model.meta["wavelengths"]),osmd.sat.shape[0],osmd.sat.shape[1]))
        for band,index in zip(osmd.model.meta["bandlist"],range(len(osmd.model.meta["wavelengths"]))):
            osmd.sat.sat_band_np[index,:,:]= osmd.sat.bands[band].values
            
        
        
        ga.obs(osmd)
        ga.refl_estimate(osmd)
        ga.ssa_estimate(osmd)#, mask_bands)
        ga.mask_estimate(osmd)
          
        #Plots in iteration 
        y=25
        x=5        
        fplot.PlotDMPix(outputs_folder,"Plat",osmd,iterOptim,x,y)#,mask_bands)
        fplot.PlotDMPixRADIANCE(outputs_folder,"Plat",osmd,iterOptim,x,y)
        fplot.PlotSsaMaskSnow(outputs_folder, osmd, iterOptim)
        fplot.PlotHistGenerateMask(outputs_folder, osmd, iterOptim)
                

    # Save final dataset
    write_xarray_dset(osmd.model.snowmask, outputs_folder+"%s/subpixelsnow.tiff"% osmd.sat.date, 
                      2154, osmd.sat.geotransform)
    write_xarray_dset(osmd.model.toa_rad, outputs_folder+"%s/toa_level.tiff" %(osmd.sat.date), 
                      2154, osmd.sat.geotransform)
