#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the complex_terrain algorithm
M. Lamare, M. Dumont, G. Picard (IGE, CEN).
"""
import numpy as np
import xarray as xr
from redress.geospatial import gdal_ops
# Import esa specific toolbox Science Toolbox Exploitation Platform
from snappy import (ProductIO, ProductUtils, HashMap, GPF, WKTReader)
from redress.inputs.products import Sat


def open_product(inpath):
    """Open a satellite product.

    Opens the satellite product based on snappy functions, if a list of
     corner coordinates is provided the reader will open a geographical
     subset based on the extent of the coordinates.

    Args:
        (self): the snappy_reader class
        epsg (int): EPSG code to reproject data.
        geo_extent (wkt): BoundingBox in WKT format

    Returns:
        (self.product): ESA SNAP product object
    """

    try:
        # Open the product
        product = ProductIO.readProduct(str(inpath))
        print(product.getSceneGeoCoding())

    except:  # Bare except to be able to catch SNAP errors (JAVA)
        print("Error: SNAP cannot read specified file!")

    return product


def image_extents(product, step=1):
    """ Get the corner coordinates from the opened satellite image.

    Fetches a list of latitude and longitude values for the boundary
    of a given satellite product with a given pixel step along
    the border.

    Args:
        (self.product): The snappy product opened using snappy.
        step (int): the step size in pixels

    Returns:
            (list): a list containing the boundary coordinates."""

    # Build the image boundary with a step of 10 pixels.
    image_boundary = ProductUtils.createGeoBoundary(product, step)

    # Create a list with the coordinate pairs (tuples)
    coord_values = []

    for coords in image_boundary:
        coord_values.append((coords.getLon(), coords.getLat()))

    return coord_values

def fetch_bandnames(product):
    """ Get the bandnames from the product.

    Fetches a list of bandnames from the opened satellite product.

    Args:
        (self.product): The snappy product opened using snappy.

    Returns:
        (list): A list of bandnames in the product.
    """
    bands = list(product.getBandNames())
    tpgs = list(product.getTiePointGridNames())
    bandlist = bands + tpgs
    
    return bandlist

def open_subset(product, geo_extent, copymetadata="true"):
    """ Create a subset of a given SNAP product.

    From a given set of coordinates defining a region of interest create
    a subset product containing all the bands from the original product.

    Args:
        inprod: ESA SNAP product object
        geo_extent (wkt): BoundingBox in WKT format
        copymetadata (bool): Copy all bands to subset product

    Returns:
        (self.product): ESA SNAP product object
    """

   


def resample(product, geo_extent, epsg, res, copymetadata="true"):
    """ Get the corner coordinates from the opened satellite image.

    Fetches a list of latitude and longitude values for the boundary
    of a given satellite product with a given pixel step along
    the border.

    Args:
        (self.product): The snappy product opened using snappy.
        step (int): the step size in pixels
        res : float, resolution of the product

    Returns:
            (list): a list containing the boundary coordinates."""
    # Step 1: reproject
    # Empty HashMap
    parameters = HashMap()

    # Snap operator parameters for reprojection
    parameters.put("addDeltaBands", "false")
    parameters.put("crs", str(epsg))
    parameters.put("resampling", "Bilinear")
    parameters.put("noDataValue", "-9999")
    parameters.put("orthorectify", "false")
    parameters.put("pixelSizeX", res)
    parameters.put("pixelSizeY", res)

    # Check if the image rasters are all the same size, if not, offer to
    # resample
    if product.isMultiSize():
        raise ValueError("Product contains bands of different sizes and"
                         " can not be processed.")
    else:
        # Call the operator depending on the resample
        reproj_prod = GPF.createProduct("Reproject", parameters,
                                        product)

    parameters = None

    # Step 2: subset to extent
    # Empty HashMap
    parameters = HashMap()

    # Subset parameters
    geo = WKTReader().read(geo_extent.ExportToWkt())
    parameters.put("geoRegion", geo)
    parameters.put("subSamplingX", "1")
    parameters.put("subSamplingY", "1")
    parameters.put("copyMetadata", copymetadata)

    # Create subset using operator
    prod_subset = GPF.createProduct("Subset", parameters, reproj_prod)

    reproj_prod = None

    return prod_subset


def extract_band(product, bandname, unit=None):
    """ Extract a band from the product.

    Extracts the band values for a given band name from the product as a
    DataArray. The units are added as an attribute.

    Args:
        bandname (str): the name of the band to extract the data from.
        unit (str): unused here, set for module compatibility.
    Returns:
        (DataArray): an xarray containing the band values and an attribute
         with the units.
    """
    # Get product size
    height = product.getSceneRasterHeight()
    width = product.getSceneRasterWidth()

    try:
        currentband = product.getBand(bandname)  # Open band
        if currentband is None:
            currentband = product.getRasterDataNode(bandname)
    except ValueError as err:
        print("Error importing band: %s! %s" % (bandname, err))
        
        
    # Get coordinates
    try:
        latband = product.getBand("latitude")  # Open band
        if latband is None:
            latband = product.getRasterDataNode("latitude")
    except ValueError as err:
        print("Error importing latitude data! %s" % (err))
        
    try:
        lonband = product.getBand("longitude")  # Open band
        if lonband is None:
            lonband = product.getRasterDataNode("longitude")
    except ValueError as err:
        print("Error importing longitude data! %s" % (err))
        
    # Initialise an empty array and read pixels
    latarray = np.zeros((height, width), dtype=np.float32)
    lat = latband.readPixels(0, 0, width, height, latarray)

    # Initialise an empty array and read pixels
    lonarray = np.zeros((height, width), dtype=np.float32)
    lon = lonband.readPixels(0, 0, width, height, lonarray)

    # Initialise an empty array and read pixels
    array = np.zeros((height, width), dtype=np.float32)
    bandraster = currentband.readPixels(0, 0, width, height, array)
    
    # Convert numpy array to xarray
    band = xr.DataArray(bandraster, dims=['y', 'x'],)
    band.attrs["units"] = currentband.getUnit()  # Set units

    band.coords['lon'] = xr.DataArray(lon, dims=['y', 'x'])
    band.coords['lat'] = xr.DataArray(lat, dims=['y', 'x'])
    return band

def get_wavelength(product, bandname):
        currentband = product.getBand(bandname)
        return currentband.getSpectralWavelength()


def get_geotransform(product):

    # Get the scene geotransform
    gt = product.getSceneGeoCoding().getImageToMapTransform().\
        toString()

    # Split to readable parameters
    for i in gt.split("\n"):
        if "elt_0_0" in i:
            elt_0_0 = i.split(",")[1][1:-2]
        elif "elt_0_2" in i:
            elt_0_2 = i.split(",")[1][1:-2]
        elif "elt_1_1" in i:
            elt_1_1 = i.split(",")[1][1:-2]
        elif "elt_1_2" in i:
            elt_1_2 = i.split(",")[1][1:-2]

    # Check for the existance of the gt parameters
    if not all([elt_0_0, elt_0_2, elt_1_1, elt_1_2]):
        raise ValueError("Geotransform data missing in product!")

    return (float(elt_0_2), float(elt_0_0), 0.0, float(elt_1_2),
            0.0, float(elt_1_1))


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
    s3_prod = open_product(inpath)

    # Check if the product overlaps the extent of the file
    s3_extents = gdal_ops.build_poly_from_coords(image_extents(s3_prod))
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

    # Resample dataset
    s3_resampled = resample(s3_prod, extent, epsg, res=300.)

    # Instantiate class
    s3_data = Sat()

    # Feed the class with data
    s3_data.meta["path"] = inpath

    # Get radiance bands from product and import to the S3 class
    for band in user_list:
        # Get radiance bands
        if "Oa" in band:
            # Fetch data from band (get firstpart from bandname)
            s3_data.bands[band.split('_')[0]] = extract_band(s3_resampled,
                                                             band,
                                                             "radiance")
            # Get band name
            s3_data.meta["bandlist"].append(band.split('_')[0])

            # Get band wavelength center
            s3_data.meta["wavelengths"].append(get_wavelength(s3_resampled,
                                                              band))
        # Get solar and viewing bands from the S3 product
        elif band in ["SZA",
                      "SAA",
                      "OZA",
                      "OAA"]:

            # Import band
            s3_data.angles[band] = extract_band(s3_resampled,
                                                band,
                                                )

    # Correct for the particularity of the VAA band in S3 OLCI
    # (flipped on one side of nadir)
    negative_array = \
        s3_data.angles["OAA"].where(
            s3_data.angles["OAA"] < 0)
    positive_array = \
        s3_data.angles["OAA"].where(
            s3_data.angles["OAA"] > 0)
    # Replace negative values by substracting 180 by the positive value
    negative_array = negative_array + 180

    # Merge arrays
    s3_data.angles["OAA"] = \
        positive_array.fillna(negative_array)

    # Rename angle bands for consistency
    s3_data.angles = s3_data.angles.rename({"OAA": "VAA",
                                            "OZA": "VZA",
                                            })
    # Convert to radians
    for band in ["SZA", "SAA", "VZA", "VAA"]:
        s3_data.angles[band].data = xr.ufuncs.deg2rad(s3_data.angles[band])
        s3_data.angles[band].attrs["units"] = "radians"
        s3_data.meta["angles"].append(band)

    # Get/Set Geotransform
    s3_data.geotransform = get_geotransform(s3_resampled)
    
    s3_resampled = None

    return s3_data
