#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gdal_reader.

This file is part of the complex_terrain algorithm
M. Lamare, M. Dumont, G. Picard (IGE, CEN).
"""
import gdal
import xarray as xr
import numpy as np
from pyproj import Proj, transform
from redress.geospatial import gdal_ops
from redress.inputs.products import Dem


def open_product(inpath):
    """Open product with GDAL.

    Open a raster and return the Gdal dataset and geotransform information.

    :param inpath: Path to the raster
    :type inpath: pathlib.PosixPath
    :return: a tuple containing the Gdal dataset and geotransform
    :rtype: tuple (osgeo.gdal.Dataset, tuple)
    """
    try:
        # Open the product
        prod = gdal.Open(str(inpath))

    except RuntimeError as err:
        print("Error: GDAL cannot read specified file! %s" % err)

    # Extract information from the product (overwrite)
    product = prod
    geotransform = prod.GetGeoTransform()

    return (product, geotransform)


def image_extents(product, geotransform, product_epsg, step=1):
    """Get the corner coordinates from the opened raster.

    Fetches a list of latitude and longitude values for the boundary
    of a given satellite product with a given pixel step along
    the border.

    :param product: the opened Gdal dataset
    :type product: osgeo.gdal.Dataset
    :param geotransform: the raster's geotransform information
    :type geotransform: tuple
    :param step: the step size in pixels
    :type step: int, optional
    :return: a list containing the boundary coordinates
    :rtype: list
    """
    # Build the projection data
    prj_in = Proj('epsg:%s' % product_epsg)
    prj_out = Proj('epsg:%s' % 4326)
    
    # Get Dem product size
    size = product.RasterXSize, product.RasterYSize

    # Set empty extent file and retrieve lat/lon values
    ext = []
    xarr = [0, size[0]]
    yarr = [0, size[1]]

    for px in xarr:
        for py in yarr:
            x = geotransform[0] + (px * geotransform[1]) +\
                (py * geotransform[2])
            y = geotransform[3] + (px * geotransform[4]) +\
                (py * geotransform[5])
            xy_dd = transform(prj_in, prj_out, x, y)
            ext.append((xy_dd[1], xy_dd[0]))
        yarr.reverse()
    
    return ext


def reproject(product, epsg):
    """Reproject a product.

    Reprojects a Gdal product in memory based on a given EPSG code. The
     operation is performed in-place.

    :param epsg: the epsg code to reproject the product to
    :type epsg: int
    """
    reprojected = gdal.Warp('/vsimem/reproj.tif', product,
                            dstSRS='EPSG:%s' % epsg, dstNodata=-9999)

    return (reprojected, reprojected.GetGeoTransform())


def clip(product, geo_extent, native_epsg=4326, epsg=4326):
    """Clip a product.

    Clips a Gdal product in memory to a given extent. The
     operation is performed in-place.

    :param geo_extent: the extent that the product will be clipped to
    :type geo_extent: osgeo.ogr.Geometry
    :param epsg: the epsg code of the extent
    :type epsg: int
    """
    # Build the projection data
    prj_in = Proj('epsg:%s' % native_epsg)
    prj_out = Proj('epsg:%s' % epsg)

    # Get max min coords of the shape
    minX, maxX, minY, maxY = geo_extent.GetEnvelope()

    # Product extent
    xmin, ymin = transform(prj_in, prj_out, minY, minX)
    xmax, ymax = transform(prj_in, prj_out, maxY, maxX)

    bbox = [xmin, ymax, xmax, ymin]

    clipped = gdal.Translate('/vsimem/clip.tif', product,
                             projWin=bbox)

    return (clipped, clipped.GetGeoTransform())


def extract_band(product, geotransform):
    """Extract a band from the product.

    Extracts the first band values from the product as a
    DataArray. The units are added as an attribute.

    :return: an xarray containing the band values.
    :rtype: xarray
    """
    try:
        currentband = product.GetRasterBand(1)  # Open band
    except ValueError as err:
        print("Error importing the raster band. %s" % err)

    # Extract band array
    arr = currentband.ReadAsArray()
    band = xr.DataArray(arr, dims=['y', 'x'])

    # Calculate longitude array
    indices = np.indices(arr.shape)
    longitude = geotransform[1] * indices[1] + geotransform[2] *\
        indices[0] + geotransform[1] * 0.5 + geotransform[2] *\
        0.5 + geotransform[0]

    # Calculate latitude array
    latitude = geotransform[4] * indices[1] + geotransform[5] *\
        indices[0] + geotransform[4] * 0.5 + geotransform[5] *\
        0.5 + geotransform[3]

    # Set to xarray
    band.coords['lon'] = xr.DataArray(longitude, dims=['y', 'x'])
    band.coords['lat'] = xr.DataArray(latitude, dims=['y', 'x'])

    return band


def dem_generic(inpath, extent, epsg,):
    """doc."""
    # Instantiate DEM class
    dem = Dem()

    # Feed the class with data
    dem.meta["path"] = inpath

    # Open the image using the imported reader
    raw_product, raw_geotrans = open_product(inpath)

    # Check if the product overlaps the extent of the file
    dem_extents = gdal_ops.build_poly_from_coords(image_extents(raw_product,
                                                                raw_geotrans,
                                                                epsg))

    if not gdal_ops.geom_contains(dem_extents, extent):
        raise ValueError("The chosen bounding box is not entirely inside the"
                         " provided DEM!")

    # Reproject the DEM to the desire dprojection
    reprojected_product, reprojected_geotrans = reproject(raw_product, 2154)

    # Subset the DEM to an area of interest
    clipped_product, clipped_geotrans = clip(reprojected_product, extent,
                                             epsg=2154)

    # Import the band and geotransform
    dem.bands["altitude"] = extract_band(clipped_product, clipped_geotrans)
    dem.geotransform = clipped_geotrans
    dem.pixel_size = (clipped_geotrans[1], clipped_geotrans[-1])

    return dem
