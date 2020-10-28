#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the complex_terrain algorithm
M. Lamare, M. Dumont, G. Picard (IGE, CEN).
"""
import numpy as np
import xarray as xr
from snappy import (ProductIO, ProductUtils, HashMap, GPF, WKTReader)


class reader(object):
    """ESA SNAP reader

    The snappy_reader calls the snappy library (python interface of the ESA
    SNAP program) to open and import satellite data.
    """
    def __init__(self):
        self.path = None
        self.product = None
        self.bandlist = list()
        self.tpg_list = list()

    def open_product(self):
        """ Opens a satellite product.

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
            prod = ProductIO.readProduct(str(self.path))

        except:  # Bare except to be able to catch SNAP errors (JAVA)
            print("Error: SNAP cannot read specified file!")

        # Update product
        self.product = prod

        # Close dataset
        prod = None

    def open_subset(self, geo_extent, copymetadata="true"):
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

        # Empty HashMap
        parameters = HashMap()

        # Subset parameters
        geo = WKTReader().read(geo_extent)
        parameters.put("geoRegion", geo)
        parameters.put("subSamplingX", "1")
        parameters.put("subSamplingY", "1")
        parameters.put("copyMetadata", copymetadata)

        # Create subset using operator
        prod_subset = GPF.createProduct("Subset", parameters, self.product)

        # Update product
        self.product = prod_subset

        # Close dataset
        prod_subset = None

    def reproject(self, epsg):
        """ Get the corner coordinates from the opened satellite image.

        Fetches a list of latitude and longitude values for the boundary
        of a given satellite product with a given pixel step along
        the border.

        Args:
            (self.product): The snappy product opened using snappy.
            step (int): the step size in pixels

        Returns:
                (list): a list containing the boundary coordinates."""
        # Empty HashMap
        parameters = HashMap()

        # Snap operator parameters for reprojection
        parameters.put("addDeltaBands", "false")
        parameters.put("crs", str(epsg))
        parameters.put("resampling", "Nearest")
        parameters.put("noDataValue", "-9999")
        parameters.put("orthorectify", "false")
        parameters.put("pixelSizeX", 300.0)
        parameters.put("pixelSizeY", 300.0)

        # Check if the image rasters are all the same size, if not, offer to
        # resample
        if self.product.isMultiSize():
            raise ValueError("Product contains bands of different sizes and"
                             " can not be "
                             "processed. Resampling neccessary so that all"
                             " rasters have the same size.")
            # TODO add an option to resample or not
        else:
            # Call the operator depending on the resample
            reprojProd = GPF.createProduct("Reproject", parameters,
                                           self.product)

        # Update product
        self.product = reprojProd

        # Close dataset
        reprojProd = None

    def fetch_bandnames(self):
        """ Get the bandnames from the product.

        Fetches a list of bandnames from the opened satellite product.

        Args:
            (self.product): The snappy product opened using snappy.

        Returns:
            (list): A list of bandnames in the product.
        """
        bands = list(self.product.getBandNames())
        tpgs = list(self.product.getTiePointGridNames())
        self.bandlist = bands + tpgs

    def extract_band(self, bandname, unit=None):
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
        height = self.product.getSceneRasterHeight()
        width = self.product.getSceneRasterWidth()

        try:
            currentband = self.product.getBand(bandname)  # Open band
            if currentband is None:
                currentband = self.product.getRasterDataNode(bandname)
        except ValueError as err:
            print("Error importing band: %s! %s" % (bandname, err))

        # Initialise an empty array and read pixels
        array = np.zeros((height, width), dtype=np.float32)
        bandraster = currentband.readPixels(0, 0, width, height, array)

        # Convert numpy array to xarray
        band = xr.DataArray(bandraster, dims=['x', 'y'])
        band.attrs["units"] = currentband.getUnit()  # Set units

        return band

    def get_wavelength(self, bandname):
        currentband = self.product.getBand(bandname)
        return currentband.getSpectralWavelength()

    def get_geotransform(self):

        # Get the scene geotransform
        gt = self.product.getSceneGeoCoding().getImageToMapTransform().\
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

    def image_extents(self, step=1):
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
        image_boundary = ProductUtils.createGeoBoundary(self.product, step)

        # Create a list with the coordinate pairs (tuples)
        coord_values = []

        for coords in image_boundary:
            coord_values.append((coords.getLon(), coords.getLat()))

        return coord_values
