#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 15:26:03 2021

@author: nheilir
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Rugged Terrain code.

This script run the rugged terrain model for testing and debugging purposes.
"""
import warnings
import sys
import os
import numpy as np

sys.path.append("/home/nheilir/REDRESS/REDRESS")
from redress.inputs.products import Model,SMD
import redress.display.mask_res_plot as fplot
import redress.optics.generic_algorithm as ga
from redress.geospatial.gdal_ops import (write_xarray_dset,write_xarray)
warnings.filterwarnings('ignore')

inputs_folder =  "Path to an input folder where dem and sat product are stocks"

Sattype="sat3 ou sat2"
demfile="nom du fichier DEM"
demname=os.path.splitext(demfile)[0]

outputs_folder = "Path to an output folder"+Sattype+demname+"/"
if not os.path.exists(outputs_folder): 
  # Create the outputs folder
  os.makedirs(outputs_folder)
  print("The new directory  %s is created!"%outputs_folder)

# index of spectral band to ignore []
mask_bands=None

# atmospherique value, the studied date must be added
measure_value = {
                 '20180213' :{'aod':0.02 , 'ozone':0.6 , 'water':0.48 , 'fixssa':41.41},
                 '20180221' :{'aod':0.03 , 'ozone':0.5 , 'water':1.5 , 'fixssa':33.66},
                 '20180314' :{'aod':0.005 , 'ozone':0.3966 , 'water':0.9 , 'fixssa':45.42},
                 '20180316' :{'aod':0.01 , 'ozone':0.6 , 'water':1.7 , 'fixssa':27.37},
                 '20180320' :{'aod':0.01 , 'ozone':0.6 , 'water':1.7 , 'fixssa':27.37},
                 '20180321' :{'aod':0.01 , 'ozone':0.6 , 'water':1.7 , 'fixssa':27.37},
                 '20180322' :{'aod':0.01 , 'ozone':0.6 , 'water':1.7 , 'fixssa':27.37},
                 '20180406' :{'aod':0.09 , 'ozone':0.347 , 'water':0.512 , 'fixssa':5.91}, 
                 '20180417' :{'aod':0.15 , 'ozone':0.08 , 'water':3. , 'fixssa':5.91}, 
                 '20180420' :{'aod':0.15 , 'ozone':0.08 , 'water':3. , 'fixssa':5.91},
                 }

osmd=SMD()

# read dem product in osmd.dem
osmd.init_DEM(outputs_folder,inputs_folder, demfile,"gdal_reader")

date="20180417"
sat_path = inputs_folder+"path of sat product :"
'''
 for sen3 path up to /xfdumanifest.xml
 for sat2 a directory above .SAFE 
'''
# create a folder for each date
if not os.path.exists(outputs_folder+date):
    os.makedirs(outputs_folder+date) 
# read sattelite product in osmd.sat
osmd.init_SAT(outputs_folder,inputs_folder, date, sat_path,"pytroll_satpy",Sattype,demname)
   
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
for band,index in zip(osmd.model.meta["bandlist"],range(len(osmd.model.meta["wavelengths"]))):
    osmd.sat.sat_band_np[index,:,:]= osmd.sat.bands[band].values    

# iteration to estimate snow mask and ssa (threshold to be defined )
for iterOptim in range(1,2):
    osmd.model.simulate_TOA_radiance(osmd.model) 

    ga.obs(osmd) #calculate rtild
    ga.refl_estimate(osmd) #calculate r
    ga.ssa_estimate(osmd, mask_bands,model="2-param") #calculate direct_model
    ga.mask_estimate(osmd)

    fplot.PlotSsaMaskSnow(outputs_folder, osmd, iterOptim)
    fplot.PlotHistGenerateMask(outputs_folder, osmd, iterOptim)
    
    # Save final dataset in tif files
    write_xarray_dset(osmd.model.snowmask, outputs_folder+"%s/subpixelsnow%s.tiff"%(osmd.sat.date,iterOptim), 
                      2154, osmd.sat.geotransform)
    write_xarray_dset(osmd.model.toa_rad, outputs_folder+"%s/toa_level%s.tiff" %(osmd.sat.date,iterOptim), 
                      2154, osmd.sat.geotransform)

    write_xarray(osmd.model.a,  outputs_folder+"%s/a%s.tiff" %(osmd.sat.date,iterOptim),2154, osmd.sat.geotransform)  
    write_xarray(osmd.model.rmse,  outputs_folder+"%s/rmse%s.tiff" %(osmd.sat.date,iterOptim),2154, osmd.sat.geotransform) 
    





