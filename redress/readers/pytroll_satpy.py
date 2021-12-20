#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pytroll_satpy.

This file is part of the complex_terrain algorithm
M. Lamare, M. Dumont, G. Picard (IGE, CEN).
"""
import numpy as np
from glob import glob
from satpy import Scene, find_files_and_readers, available_readers
import netCDF4
import xarray as xr
from pyresample import create_area_def
from pyproj import Proj, transform
from redress.geospatial import gdal_ops
from redress.inputs.products import Sat
import matplotlib.pyplot as plt
import lxml.etree as ET

def open_generic_product(inpath,reader):
    """Open a satellite product.

    Open a satellite product based on satpy functions without specifying a
    reader.

    :param inpath: Path to the folder containing the satellite images
    :type inpath: pathlib.PosixPath
    :return: Satpy scene containing the opened satellite product
    :rtype: satpy.scene.Scene
    """
    # Get files in the satellite folder
    if reader=='olci_l1b':
        fpath_nc = inpath.parents[0].joinpath('*')
        fnames = glob(str(fpath_nc))
    elif reader=='msi_safe':
        fnames = find_files_and_readers(base_dir=str(inpath),reader='msi_safe')

    # Open product and store
    prod = Scene(filenames=fnames, reader=reader)

    return prod


def fetch_bandnames(product):
    """Get the bandnames from the open product.

    Fetches a list of bandnames from the opened satpy scene.

    :param product: Satpy Scene
    :type product: satpy.scene.Scene
    :return: List of all product bands
    :rtype: list
    """
    return product.all_dataset_names()


def image_extents(product, epsg=4326, native_epsg=4326, calibration=None):
    """Get the corner coordinates from the opened satellite image.

    Fetches the corner coordinates from the product in the specified
     projection based on an EPSG code.

    :param product: Satpy Scene
    :type product: satpy.scene.Scene
    :param epsg: EPSG code of the projection of the coordinates
    :type epsg: int, optional
    :param native_epsg: EPSG code of the product (default to 4326)
    :type native_epsg: int, optional
    :return: four corner coordinate tuples
    :rtype: list [4]
    """
    # Build the projection data
    prj_in = Proj('epsg:%s' % native_epsg)
    prj_out = Proj('epsg:%s' % epsg)

    # Get latitude and longitude from the product
    try:
        # Search for inbeddded data
        bandlist = fetch_bandnames(product)
        if not bandlist:
            bandlist = product.keys()
        product.load([bandlist[0]], calibration)
        lats, lons = product[bandlist[0]].attrs[
            'area'].get_lonlats()
    except ValueError as err:
        print("Lat/lon data not found, looking in band data.", err)
        try:
            # Search for bands
            product.load(["latitude", "longitude"])
            lats = product["latitude"].compute().data
            lons = product["longitude"].compute().data

        except ValueError as err:
            print("No lat/lon data found in product.", err)

    # Product extent
    xmin, ymin = transform(prj_in, prj_out, np.min(lats), np.min(lons))
    xmax, ymax = transform(prj_in, prj_out, np.max(lats), np.max(lons))
    return [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]


def sentinel3_olci(inpath, extent, epsg, user_list=[], resolution=300,):
    """Read an S3 OLCI image.

    Open a Sentinel-3 OLCI image based on the path to the SEN3 folder. The
    image is cropped to the extent provided as an input and reprojected to the
    given EPSG code. The user can specify a list of bands to process and a
    resolution.

    :param inpath: Path to the Sentinel-3 OLCI image
    :type inpath: pathlib.PosixPath
    :param extent: Polygon setting the extent of the Scene to process
    :type extent: osgeo.ogr.Geometry
    :param epsg: EPSG code of the projection for the image
    :type epsg: int
    :param user_list: List of bandnames to process
    :type user_list: list, optional
    :param resolution: Resolution of the image to resample the image to
    :type resolution: int, optional
    :return: The satellite data stored in a class
    :rtype: redress.inputs.products.Sat
    """
    # Open the image
    s3_prod = open_generic_product(inpath,reader='olci_l1b')
    # Don't know how to open the following files: {'/home/nheilir/REDRESS/REDRESS_files/Entree_modele/S3A_OL_1_EFR____20180417T094642_20180417T094942_20180418T134454_0179_030_136_2160_LN1_O_NT_002.SEN3/removed_pixels.nc', '/home/nheilir/REDRESS/REDRESS_files/Entree_modele/S3A_OL_1_EFR____20180417T094642_20180417T094942_20180418T134454_0179_030_136_2160_LN1_O_NT_002.SEN3/xfdumanifest.xml', '/home/nheilir/REDRESS/REDRESS_files/Entree_modele/S3A_OL_1_EFR____20180417T094642_20180417T094942_20180418T134454_0179_030_136_2160_LN1_O_NT_002.SEN3/time_coordinates.nc', '/home/nheilir/REDRESS/REDRESS_files/Entree_modele/S3A_OL_1_EFR____20180417T094642_20180417T094942_20180418T134454_0179_030_136_2160_LN1_O_NT_002.SEN3/tie_geo_coordinates.nc', '/home/nheilir/REDRESS/REDRESS_files/Entree_modele/S3A_OL_1_EFR____20180417T094642_20180417T094942_20180418T134454_0179_030_136_2160_LN1_O_NT_002.SEN3/qualityFlags.nc'}

    # Check if the product overlaps the extent of the file
    s3_extents = gdal_ops.build_poly_from_coords(image_extents(s3_prod,
                                                               calibration="radiance"))
    if not gdal_ops.geom_contains(s3_extents, extent):
        raise ValueError("The chosen bounding box is not entirely inside the"
                         " provided Satellite image!")

    # If the user selects all bands, set a list containing all band names
    if not user_list:
        user_list = fetch_bandnames(s3_prod)
    else:
        # Check if bands are in the product
        if not set(user_list).issubset(fetch_bandnames(s3_prod)):
            raise ValueError("Selected bands are not in product!")

    # Load all bands for the resampling
    for band in user_list:
        if band.startswith("Oa"):
            s3_prod.load([band], calibration="radiance")
        else:
            s3_prod.load([band])

    # Set parameters for the resampling stage
    image_extent = gdal_ops.corner_coords_from_poly(extent, out_epsg=2154)
    area_def = create_area_def("reprojection", "EPSG:%s" % epsg,
                               area_extent=image_extent,
                               units="metres", resolution=300,
                               )

    # Resample product
    s3_resampled = s3_prod.resample(area_def, resampler="nearest",res=300.)

    # Instantiate class
    s3_data = Sat()

    # Feed the class with data
    s3_data.meta["path"] = inpath

    # Get radiance bands from product and import to the S3 class
    for band in user_list:
        # Get radiance bands
        if "Oa" in band:
            # Load radiance band
            s3_resampled.load([band], calibration='radiance')

            # Fetch data from band
            s3_data.bands[band] = s3_resampled[band]

            # Get band name
            s3_data.meta["bandlist"].append(band)

            # Get band wavelength center
            s3_data.meta["wavelengths"].append(
                s3_resampled[band].wavelength[1] * 1000)

        # Get solar and viewing bands from the S3 product
        elif band in ["satellite_azimuth_angle",
                      "satellite_zenith_angle",
                      "solar_azimuth_angle",
                      "solar_zenith_angle"]:

            # Load and import band
            s3_resampled.load([band])
            s3_data.angles[band] = s3_resampled[band]

    # Correct for the particularity of the VAA band in S3 OLCI
    # (flipped on one side of nadir)
    negative_array = \
        s3_data.angles["satellite_azimuth_angle"].where(
            s3_data.angles["satellite_azimuth_angle"] < 0)
    positive_array = \
        s3_data.angles["satellite_azimuth_angle"].where(
            s3_data.angles["satellite_azimuth_angle"] > 0)
    # Replace negative values by substracting 180 by the positive value
    negative_array = negative_array + 180

    # Merge arrays
    s3_data.angles["satellite_azimuth_angle"] = \
        positive_array.fillna(negative_array)

    # Rename angle bands for consistency
    s3_data.angles = s3_data.angles.rename({"satellite_azimuth_angle": "VAA",
                                            "satellite_zenith_angle": "VZA",
                                            "solar_azimuth_angle": "SAA",
                                            "solar_zenith_angle": "SZA"},
                                           )
    # Convert to radians
    for band in ["SZA", "SAA", "VZA", "VAA"]:
        s3_data.angles[band].data = xr.ufuncs.deg2rad(s3_data.angles[band])
        s3_data.angles[band].attrs["units"] = "radians"
        s3_data.meta["angles"].append(band)

    # Get/Set Geotransform
    nx = len(s3_data.bands.coords['x'])
    ny = len(s3_data.bands.coords['y'])
    xmin, ymin, xmax, ymax = [float(s3_data.bands.coords['x'].min().values),
                              float(s3_data.bands.coords['y'].min().values),
                              float(s3_data.bands.coords['x'].max().values),
                              float(s3_data.bands.coords['y'].max().values)]
    xres = (xmax - xmin) / float(nx)
    yres = (ymax - ymin) / float(ny)
    s3_data.geotransform = (xmin, xres, 0, ymax, 0, -yres)

    return s3_data



def sentinel2_msi_safe(inpath, extent, epsg, user_list=[], resolution=5,):
    """Read an S2 L1C image.

    Open a Sentinel-2 L1C image based on the path to the SEN2 folder. The
    image is cropped to the extent provided as an input and reprojected to the
    given EPSG code. The user can specify a list of bands to process and a
    resolution.

    :param inpath: Path to the Sentinel-2 MSI image
    :type inpath: pathlib.PosixPath
    :param extent: Polygon setting the extent of the Scene to process
    :type extent: osgeo.ogr.Geometry
    :param epsg: EPSG code of the projection for the image
    :type epsg: int
    :param user_list: List of bandnames to process
    :type user_list: list, optional
    :param resolution: Resolution of the image to resample the image to
    :type resolution: int, optional
    :return: The satellite data stored in a class
    :rtype: redress.inputs.products.Sat
    """
    # Open the image
    s2_prod = open_generic_product(inpath,reader='msi_safe')

    # Check if the product overlaps the extent of the file
    s2_extents = gdal_ops.build_poly_from_coords(image_extents(s2_prod))
    if not gdal_ops.geom_contains(s2_extents, extent):
        raise ValueError("The chosen bounding box is not entirely inside the"
                         " provided Satellite image!")
    #create solar_irradiance dictionary
    
    all_bands = fetch_bandnames(s2_prod)
    solar_irradiance=get_meta_solar_irradiance(inpath,all_bands)
    # If the user selects all bands, set a list containing all band names
    if not user_list:
        user_list = fetch_bandnames(s2_prod)
    else:
        # Check if bands are in the product
        if not set(user_list).issubset(fetch_bandnames(s2_prod)):
            raise ValueError("Selected bands are not in product!")
    # Load all bands for the resampling
    for band in user_list:
        if band.startswith("B"):
            s2_prod.load([band])
        else:
            s2_prod.load([band])

    # Set parameters for the resampling stage
    image_extent = gdal_ops.corner_coords_from_poly(extent, out_epsg=2154)
    area_def = create_area_def("reprojection", "EPSG:%s" % epsg,
                               area_extent=image_extent,
                               units="metres", resolution=20,#dem resolution
                               )

    # Resample product
    s2_resampled = s2_prod.resample(area_def, resampler="nearest",res=20.)

    # Instantiate class
    s2_data = Sat()

    # Feed the class with data
    s2_data.meta["path"] = inpath
    s2_data.meta["solar_irradiance"]=solar_irradiance
    # Get radiance bands from product and import to the S3 class
    for band in user_list:#in range(len(user_list)):
        
#        band=user_list[band_index]
        
        # Get radiance bands
        if "B" in band:
            # Load radiance band
            s2_resampled.load([band])
            # Fetch data from band
            s2_data.bands[band] = s2_resampled[band]

            # Get band name
            s2_data.meta["bandlist"].append(band)

            # Get band wavelength center
            s2_data.meta["wavelengths"].append(
                s2_resampled[band].wavelength[1] * 1000)

        # Get solar and viewing bands from the S3 product
        elif band in ["satellite_azimuth_angle",
                      "satellite_zenith_angle",
                      "solar_azimuth_angle",
                      "solar_zenith_angle"]:

            # Load and import band
            s2_resampled.load([band])
            s2_data.angles[band] = s2_resampled[band]

    # Correct for the particularity of the VAA band in S3 OLCI
    # (flipped on one side of nadir)
    negative_array = \
        s2_data.angles["satellite_azimuth_angle"].where(
            s2_data.angles["satellite_azimuth_angle"] < 0)
    positive_array = \
        s2_data.angles["satellite_azimuth_angle"].where(
            s2_data.angles["satellite_azimuth_angle"] > 0)
    # Replace negative values by substracting 180 by the positive value
    negative_array = negative_array + 180

    # Merge arrays
    s2_data.angles["satellite_azimuth_angle"] = \
        positive_array.fillna(negative_array)
        
    # Rename angle bands for consistency
    s2_data.angles = s2_data.angles.rename({"satellite_azimuth_angle": "VAA",
                                            "satellite_zenith_angle": "VZA",
                                            "solar_azimuth_angle": "SAA",
                                            "solar_zenith_angle": "SZA"},
                                           )
    # Convert to radians
    for band in ["SZA", "SAA", "VZA", "VAA"]:
        s2_data.angles[band].data = xr.ufuncs.deg2rad(s2_data.angles[band])
        s2_data.angles[band].attrs["units"] = "radians"
        s2_data.meta["angles"].append(band)

    # Get/Set Geotransform
    nx = len(s2_data.bands.coords['x'])
    ny = len(s2_data.bands.coords['y'])
    xmin, ymin, xmax, ymax = [float(s2_data.bands.coords['x'].min().values),
                              float(s2_data.bands.coords['y'].min().values),
                              float(s2_data.bands.coords['x'].max().values),
                              float(s2_data.bands.coords['y'].max().values)]
    xres = (xmax - xmin) / float(nx)
    yres = (ymax - ymin) / float(ny)
    s2_data.geotransform = (xmin, xres, 0, ymax, 0, -yres)
    
   #projection des bande à 10 metre(B02,B03,B04,B08) sur celles de 20 metre (B05)
    for band in ["B02","B03","B04","B08"]:#,"B01", "B09", "B10"]:
        s2_data.bands[band].values=gdal_ops.resample_raster(s2_data.bands[band].values, s2_data.geotransform, s2_data.bands["B05"],
                        s2_data.geotransform, "NearestNeighbor", epsg=epsg)

    return s2_data

def get_meta_solar_irradiance(s2_str,all_bands):
    gdalFile = glob(str(s2_str)+"/*.SAFE/MTD*.xml")[0]
    xml_Scene_root=ET.parse(gdalFile).getroot()
    si=[float(ele.text) for ele in xml_Scene_root.findall(".//SOLAR_IRRADIANCE")]
    res = {all_bands[i]: si[i] for i in range(len(si))}
#    Gains = [float(ele.text) for ele in xml_Scene_root.findall(".//PHYSICAL_GAINS")][band]
    return res

