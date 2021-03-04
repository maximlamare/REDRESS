#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 08:02:21 2019

@author: rus
"""
from osgeo import gdal, gdalconst
import pyproj
from numpy import nan, sqrt
import pandas as pd


def getdata(geotif):
    """Get model or sat data."""
    # Initialise dict to store data
    outprod = {}

    # Open all bands with gdal and save to dictionnary
    ds = gdal.Open(str(geotif))
    for band in range(ds.RasterCount):
        band += 1
        srcband = ds.GetRasterBand(band)
        srcbandname = srcband.GetDescription()
        arr = srcband.ReadAsArray()
        outprod.update({srcbandname: arr})

    arr = srcband = None

    return outprod, ds


def open_sat_model(basedir):
    """Open satellite and model outputs.

    Given a folder for a date, open the satellite image and the correspoonding
    model outputs.
    """

    # Empty dictionnaries for satellite and model
    model = {}
    sats = {}

    # Open with gdal each file
    for case in basedir.iterdir():
        if ".xml" not in case.name:
            if ".tiff" in case.name:                
                if "toa" in case.name:
                    out, ds = getdata(case)
                    model.update(
                        {case.stem: out,
                         "proj": ds.GetProjection(),
                         "geotrans": ds.GetGeoTransform()})
                elif "sat" in case.name:
                    out, ds = getdata(case)
                    sats.update(
                        {case.stem: out,
                         "proj": ds.GetProjection(),
                         "geotrans": ds.GetGeoTransform()})

    return model, sats


def coordinates2xy(geotrans, lat, lon):
    """Convert a lat lon to a pixel position for an image in EPSG:2154"""

    # Get geotransform parameters
    pixel_size = geotrans[1]
    lon_origin = geotrans[0]
    lat_origin = geotrans[3]

    # Convert the lat lon values to map coordinates
    in_proj = pyproj.Proj(init="epsg:4326")
    out_proj = pyproj.Proj(init="epsg:2154")

    n_lon, n_lat = pyproj.transform(in_proj, out_proj, lon, lat)

    col = int((n_lon - lon_origin) / pixel_size)
    row = int((lat_origin - n_lat) / pixel_size)

    return row, col


def transpt(transform, coordlist):
    """Convert a list of coords to a list of pixel positions"""

    pixelcoords = [coordlist[0], coordinates2xy(transform, coordlist[1],
                                                coordlist[2])[0],
                   coordinates2xy(transform, coordlist[1],
                                  coordlist[2])[1]]
    return pixelcoords

def save_geotiff(outpath, raster, projection, geotransform):
    dst = gdal.GetDriverByName('GTiff').Create(outpath, raster.shape[1],
                                               raster.shape[0], 1,
                                               gdalconst.GDT_Float32)
    print(outpath)
    dst.SetGeoTransform(geotransform)
    dst.SetProjection(projection)

    rasterband = dst.GetRasterBand(1)
    rasterband.SetNoDataValue(nan)
    rasterband.WriteArray(raster)
    rasterband = None
    dst = None


def collocate_rasters(master, slave, output):
    # Source
    src_filename = slave
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()

    # We want a section of source that matches this:
    match_filename = master
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize

    # Output / destination
    dst_filename = output
    dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 1,
                                               gdalconst.GDT_Float32)
    dst.SetGeoTransform(match_geotrans)
    dst.SetProjection(match_proj)

    # Do the work
    gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_Average)

    del dst  # Flush


def rmse(predictions, targets):
    return sqrt(((predictions - targets) ** 2).mean())

def extract_spectra(basedir, coords):
    
    cases, sats = open_sat_model(basedir)
    angles, ds_angles = getdata(basedir.joinpath("angles.tiff"))
    topobands, ds_topo = getdata(basedir.joinpath("topobands.tiff"))   

    all_points = {}
    topo = {}
    for crd in coords:
        pixelcoords = transpt(sats['geotrans'], crd)
        pixdem = transpt(ds_angles.GetGeoTransform(), crd)
    
        # Get the individual parts from cases
        latm = {}
        lhP = {}
        ldP = {}
        l1total = {}
        l2total = {}
        l3total = {}
        l4total = {}
        l5total = {}
        sat = {}
        
        for key in cases['toa_level_1'].keys():
            if "LtA" in key:
                latm.update({key.split('_')[0]: cases['toa_level_1'][key][pixelcoords[1], pixelcoords[2]]})
            elif key.endswith('radiance'):
                l1total.update({key.split('_')[0]: cases['toa_level_1'][key][pixelcoords[1], pixelcoords[2]]})
                
        for key in cases['toa_level_2'].keys():
            if "LdP" in key:
                ldP.update({key.split('_')[0]: cases['toa_level_2'][key][pixelcoords[1], pixelcoords[2]]})
            elif "LhP" in key:
                lhP.update({key.split('_')[0]: cases['toa_level_2'][key][pixelcoords[1], pixelcoords[2]]})
            elif key.endswith('radiance'):
                l2total.update({key.split('_')[0]: cases['toa_level_2'][key][pixelcoords[1], pixelcoords[2]]})
        
        for key in cases['toa_level_3'].keys():
            if key.endswith('radiance'):
                l3total.update({key.split('_')[0]: cases['toa_level_3'][key][pixelcoords[1], pixelcoords[2]]})
        
        for key in cases['toa_level_4'].keys():
            if key.endswith('radiance'):
                l4total.update({key.split('_')[0]: cases['toa_level_4'][key][pixelcoords[1], pixelcoords[2]]})
                
        for key in cases['toa_level_5'].keys():
            if key.endswith('radiance'):
                l5total.update({key.split('_')[0]: cases['toa_level_5'][key][pixelcoords[1], pixelcoords[2]]})
    
        for key in sats['sat_bands'].keys():
            sat.update({key.split('_')[0]: sats['sat_bands'][key][pixelcoords[1], pixelcoords[2]]})
        
        topo[crd[0]] = {"SZA": angles["SZA"][pixdem[1], pixdem[2]],
                        "VZA": angles["VZA"][pixdem[1], pixdem[2]],
                        "SAA": angles["SAA"][pixdem[1], pixdem[2]],
                        "VAA": angles["VAA"][pixdem[1], pixdem[2]],
                        "slope": topobands["slope"][pixdem[1], pixdem[2]],
                        "aspect": topobands["aspect"][pixdem[1], pixdem[2]]}

        # Convert to dataframe
        all_dics = {"latm":latm, "lhP": lhP, "ldP": ldP, "case1": l1total,
                    "case2": l2total,
                    "case3": l3total, "case4": l4total, "case5": l5total,
                    "sat":sat}
    
        all_data  = pd.DataFrame.from_dict(all_dics,orient='index').T
    
        # Create wavelengths
        wvl = [400, 412.5, 442.5, 490, 510, 560, 620, 665, 673.75, 681.25,
               708.75,
               753.75, 761.25, 764.375, 767.5, 778.75, 865, 885, 900, 940,
               1020]
        wvl_nobands = [400.0, 412.5, 442.5, 490.0, 510.0, 560.0, 620.0, 665.0,
                       673.75, 681.25, 708.75, 753.75, 778.75, 865.0, 900.0,
                       1020.0]
        all_data.drop(index=["Oa13", "Oa14", "Oa15", "Oa18", "Oa20"],
                      inplace=True)
        all_data["wvl"] = wvl_nobands
    
        # Prepare the data
        all_data["case3_noatm"] = all_data["case3"] - all_data["latm"]
        all_data["case4_noatm"] = all_data["case4"] - all_data["latm"]
        all_data["case5_noatm"] = all_data["case5"] - all_data["latm"]
        
        all_points[crd[0]] = all_data
        
    return all_points, wvl, topo


def extract_spectra_new(basedir, coords):
    
    cases, sats = open_sat_model(basedir)
    angles, ds_angles = getdata(basedir.joinpath("angles.tiff"))
    topobands, ds_topo = getdata(basedir.joinpath("topobands.tiff"))   

    all_points = {}
    topo = {}
    for crd in coords:
        pixelcoords = transpt(sats['geotrans'], crd)
        pixdem = transpt(ds_angles.GetGeoTransform(), crd)
    
        # Get the individual parts from cases
        l1total = {}
        l2total = {}
        l3total = {}
        l4total = {}
        l5total = {}
        sat = {}
        
        for key in cases['toa_level_1'].keys():
            if key.startswith('Oa'):
                l1total.update({key.split('_')[0]: cases['toa_level_1'][key][pixelcoords[1], pixelcoords[2]]})
                
        for key in cases['toa_level_2'].keys():
            if key.startswith('Oa'):
                l2total.update({key.split('_')[0]: cases['toa_level_2'][key][pixelcoords[1], pixelcoords[2]]})
        
        for key in cases['toa_level_3'].keys():
           if key.startswith('Oa'):
                l3total.update({key.split('_')[0]: cases['toa_level_3'][key][pixelcoords[1], pixelcoords[2]]})
        
        for key in cases['toa_level_4'].keys():
            if key.startswith('Oa'):
                l4total.update({key.split('_')[0]: cases['toa_level_4'][key][pixelcoords[1], pixelcoords[2]]})
                
        for key in cases['toa_level_5'].keys():
            if key.startswith('Oa'):
                l5total.update({key.split('_')[0]: cases['toa_level_5'][key][pixelcoords[1], pixelcoords[2]]})
    
        for key in sats['sat_bands'].keys():
            sat.update({key.split('_')[0]: sats['sat_bands'][key][pixelcoords[1], pixelcoords[2]]})
        
        topo[crd[0]] = {"SZA": angles["SZA"][pixdem[1], pixdem[2]],
                        "VZA": angles["VZA"][pixdem[1], pixdem[2]],
                        "SAA": angles["SAA"][pixdem[1], pixdem[2]],
                        "VAA": angles["VAA"][pixdem[1], pixdem[2]],
                        "slope": topobands["slope"][pixdem[1], pixdem[2]],
                        "aspect": topobands["aspect"][pixdem[1], pixdem[2]]}

        # Convert to dataframe
        all_dics = {"case1": l1total,
                    "case2": l2total,
                    "case3": l3total, "case4": l4total, "case5": l5total,
                    "sat":sat}
    
        all_data  = pd.DataFrame.from_dict(all_dics,orient='index').T
    
        # Create wavelengths
        wvl = [400, 412.5, 442.5, 490, 510, 560, 620, 665, 673.75, 681.25,
               708.75,
               753.75, 761.25, 764.375, 767.5, 778.75, 865, 885, 900, 940,
               1020]
        wvl_nobands = [400.0, 412.5, 442.5, 490.0, 510.0, 560.0, 620.0, 665.0,
                       673.75, 681.25, 708.75, 753.75, 778.75, 865.0, 900.0,
                       1020.0]
        all_data.drop(index=["Oa13", "Oa14", "Oa15", "Oa18", "Oa20"],
                      inplace=True)
        all_data["wvl"] = wvl_nobands
    
        # Prepare the data
       
        all_points[crd[0]] = all_data
        
    return all_points, wvl, topo