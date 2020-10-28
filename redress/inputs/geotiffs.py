#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""geotiffs.

This file is part of the complex_terrain algorithm
M. Lamare, M. Dumont, G. Picard (IGE, CEN).
"""
from .satellites import import_reader


def import_DEM_gtiff(inpath, extent, epsg=None, reader=None):
    """Import a single band DEM.

    Import a DEM product using a specified reader.
    The function populates a class with the DEM altitude data and the geocoding
    bands.

    :param inpath: Path to the Sentinel-3 OLCI image
    :type inpath: pathlib.PosixPath
    :param extent: Polygon setting the extent of the Scene to process
    :type extent: osgeo.ogr.Geometry
    :param epsg: EPSG code of the projection for the image
    :type epsg: int
    :return: The DEM data stored in a class
    :rtype: redress.inputs.products.Dem
    """
    # Try to import reader library
    if isinstance(reader, str):
        dem_reader = import_reader(reader)
    else:
        raise ValueError("Please specify a valid reader.")

    # Open the image with the reader and import data to a class
    dem_prod = dem_reader.dem_generic(inpath, extent, epsg,)

    return dem_prod
