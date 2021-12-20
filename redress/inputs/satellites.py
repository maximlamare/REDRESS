#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""satellites.

This file is part of the REDRESS algorithm
M. Lamare, M. Dumont, G. Picard (IGE, CEN).
"""
import importlib
import xarray as xr
from redress.geospatial import gdal_ops
from osgeo import gdal


def import_reader(reader):
    """Import the reader backend.

    Import a reader library from the available libraries stored in the
     "readers". This function allows modularity by enabling to switch library
      to import the satellite images. Currently "esa_snap" and "pytroll_satpy"
      are implemented. Please refer to the Readme for more information on how
      to create a reader. Currently the pysat and snappy libraries are not
      compatible, so one or the other have to be installed in the conda
      environement. Snappy should be renamed in the upcoming releases:
      (https://step.esa.int/main/toolboxes/snap/)

    :param reader: Name of the library to try to import
    :type reader: str
    :return: Satellite reader class located in the readers
    :rtype: class
    """
    # Try to import the reader located in the readers folder of the project.
    # Return the reader class in the module.
    try:
        module = importlib.import_module("redress.readers.%s" % reader)
    except ImportError as e:
        raise IOError("Unable to find the reader: %s" % e)

    return module


def import_s3OLCI(inpath, extent, epsg=2154, reader=None, user_list=[]):
    """Import Sentinel-3 image.

    Import a Sentinel-3 OLCI image from the .xml file in the product folder.
    The function populates a class with the necessary information to run the
     model. The reader argument specifies the libraries used to read the
     images and can be swapped out by your own methods.

    Args:
        inpath (str): Path to a S3 OLCI image xfdumanisfest.xml file
        extent (osgeo.ogr.Geometry): GDAL geometry object specifying the extent
            to be extracted from the image
        reader (str): specify the reader to be used to deal with the image

    Returns:
        (obj): Sentinel-3 OLCI image class
    """
    # Try to import reader library
    if isinstance(reader, str):
        sat_reader = import_reader(reader)
    else:
        raise ValueError("Please specify a valid satellite reader.")

    # Open the image with the reader and import data to a class
    s3_prod = sat_reader.sentinel3_olci(inpath, extent, epsg,
                                        user_list=user_list)

    return s3_prod


def import_s2MSI_SAFE(inpath, extent, epsg=2154, reader=None, user_list=[]):
    """Import Sentinel-3 image.

    Import a Sentinel-2 SAFE image from the product folder.
    The function populates a class with the necessary information to run the
     model. The reader argument specifies the libraries used to read the
     images and can be swapped out by your own methods.

    Args:
        inpath (str): Path to a S3 OLCI image xfdumanisfest.xml file
        extent (osgeo.ogr.Geometry): GDAL geometry object specifying the extent
            to be extracted from the image
        reader (str): specify the reader to be used to deal with the image

    Returns:
        (obj): Sentinel-2 image class
    """
    # Try to import reader library
    if isinstance(reader, str):
        sat_reader = import_reader(reader)
    else:
        raise ValueError("Please specify a valid satellite reader.")

    # Open the image with the reader and import data to a class
    s2_prod = sat_reader.sentinel2_msi_safe(inpath, extent, epsg,
                                        user_list=user_list)

    return s2_prod