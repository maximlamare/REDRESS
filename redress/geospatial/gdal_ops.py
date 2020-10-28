#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gdal tools.

This file is part of the REDRESS algorithm
M. Lamare, M. Dumont, G. Picard (IGE, CEN).
"""
import json
import xarray as xr
from numpy import nan, pad
from osgeo import gdal, ogr, osr, gdalconst


def build_poly_from_coords(coord_list):
    """Build an osgeo.ogr polygon object.

    Build an osgeo.ogr geometry based on a list of coordinates.

    :param coord_list: List of input coordinates tuples (lat, lon) in
        decimal degrees (EPSG:4326)
    :type coord_list: list
    :return: GDAL geometry object, i.e. an ogr polygon
    :rtype: osgeo.ogr.Geometry
    """
    # Create an osgeo.ogr polygon using the list of coordinates
    ring = ogr.Geometry(ogr.wkbLinearRing)

    # Add the points formed by the coordinate tuples
    for coord in coord_list:
        ring.AddPoint(coord[0], coord[1])

    # Add first point to close ring
    ring.AddPoint(coord_list[0][0], coord_list[0][1])

    # Create polygon and add the created feature
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    # Set projection, hard-coded since the coordinates are in degrees
    out_srs = ogr.osr.SpatialReference()
    out_srs.ImportFromEPSG(4326)

    # Assign projection to geometry
    poly.AssignSpatialReference(out_srs)

    return poly


def build_poly_from_geojson(inpath):
    """Import a GEOJSON file with gdal.

    Import a GEOJSON, containing a single feature that represents the bounding
    box specified by the user.

    :param inpath: Path to a geojson file
    :type inpath: str
    :return: GDAL geometry object, i.e. an ogr polygon
    :rtype: osgeo.ogr.Geometry
    """
    # Read the file using osgeo.ogr and extract 1st object
    reader = ogr.Open(inpath)
    layer = reader.GetLayer()
    feature = layer.GetFeature(0)

    # Read the geojson string with json
    geom = json.loads(feature.ExportToJson())['geometry']

    # Convert into a polygon
    json_poly = build_poly_from_coords(geom['coordinates'][0])

    return json_poly


def convert_point(x, y, in_epsg, out_epsg):
    """Convert coordinates of a point.

    Convert the x and y coordinates of a point from a projection to an other.

    :param x: longitude of the point
    :type x: float
    :param y: latitude of the point
    :type y: float
    :param in_epsg: EPSG code of the input coordinates
    :type in_epsg: int
    :param out_epsg: EPSG code of the projection to be used for the output
    :type out_epsg: int
    :return: converted coordinates in the projection of the new EPSG
    :rtype: tuple
    """
    # TODO check rtype in docstring!
    # Create input / output spatial references
    in_ref = osr.SpatialReference()
    in_ref.ImportFromEPSG(in_epsg)
    out_ref = osr.SpatialReference()
    out_ref.ImportFromEPSG(out_epsg)

    # Coordinate transformation
    ct = osr.CoordinateTransformation(in_ref, out_ref)

    return ct.TransformPoint(y, x)


def corner_coords_from_poly(poly, in_epsg=4326, out_epsg=4326):
    """Convert polygon extent to min/max lat/lon.

    Fetch the bounding box of an ogr polygon, convert to a given projection,
    and return min and max latitude/longitude in the new projection.

    :param poly: OGR polygon
    :type poly: osgeo.ogr.Geometry
    :param in_epsg: EPSG code of the polygon
    :type in_epgs: int, optional
    :param out_epsg: EPSG code of the output coordinates
    :type out_epsg: int, optional
    :return: a tuple of the min and max coordinates of the polygon
    :rtype: tuple
    """
    # Get the coordinates from the bounding box
    bbox = poly.GetEnvelope()

    # Convert corners to the output epsg
    p1 = convert_point(bbox[0], bbox[2], in_epsg, out_epsg)
    p2 = convert_point(bbox[1], bbox[3], in_epsg, out_epsg)

    # Return min x, min y, max x, max y
    return (p1[0], p1[1], p2[0], p2[1])


def geom_contains(master_geom, slave_geom):
    """Check if a osgeo.ogr geometry contains an other.

    Compare two osgeo.ogr geometries, and return True or False depending on the
    fact that the 1st geomtry contains the second one or not. True is returned
    if the "master" contains the "slave".

    :param master_geom: Master GDAL geometry object
    :type master_geom: osgeo.ogr.Geometry
    :param slave_geom: Slave GDAL geometry object
    :type slave_geom: osgeo.ogr.Geometry
    :return: a boolean: True if master constains slave, False otherwise
    :rtype: bool
    """
    if master_geom.Contains(slave_geom):
        within = True
    else:
        within = False

    return within


def resample_raster(slave_raster, slave_geotrans, master_raster,
                    master_geotrans, interp_method, epsg=2154):
    """Resample a raster to match another raster's grid.

    Use GDAL to "collocate" a "slave" raster to a "master", using the rasters'
     spatial information.

    :param slave_raster: data to be regridded
    :type slave_raster: ndarray
    :param slave_geotrans: the geotransform information describing the
            slave raster
    :type slave_geotrans: list
    :param master_raster: data grid serving as reference
    :type master_raster: ndarray
    :param master_geotrans: the geotransform information describing the
            master raster
    :type master_geotrans: list
    :param interp_method: a gdal interpolation method:
            https://gdal.org/java/org/gdal/gdalconst/gdalconstConstants.html
    :type interp_method: str
    :param epsg: Slave GDAL geometry object
    :type epsg: osgeo.ogr.Geometry, optional
    :return: reinterpolated raster
    :rtype: ndarray
    """
    # Get geometry information from the master and slave
    slave_x_pixels = slave_raster.shape[1]  # number of pixels in x
    slave_y_pixels = slave_raster.shape[0]  # number of pixels in y
    master_x_pixels = master_raster.shape[1]  # number of pixels in x
    master_y_pixels = master_raster.shape[0]  # number of pixels in y

    # Build input using a virtual file
    in_driver = gdal.GetDriverByName("MEM")
    in_dataset = in_driver.Create("",
                                  slave_x_pixels,
                                  slave_y_pixels, 1,
                                  gdalconst.GDT_Float32)
    in_dataset.SetGeoTransform(slave_geotrans)

    # Set projection
    in_srs = osr.SpatialReference()
    in_srs.ImportFromEPSG(epsg)
    in_dataset.SetProjection(in_srs.ExportToWkt())

    # Set data
    in_dataset.GetRasterBand(1).WriteArray(slave_raster)

    # Build output using a virtual file
    dst = gdal.GetDriverByName("MEM")
    output = dst.Create("",
                        master_x_pixels,
                        master_y_pixels, 1,
                        gdalconst.GDT_Float32)
    output.SetGeoTransform(master_geotrans)

    # Set projection keeping the same one
    output.SetProjection(in_srs.ExportToWkt())
    b = output.GetRasterBand(1)
    b.SetNoDataValue(nan)

    # Do the work depending on the interpolation method
    if interp_method == "NearestNeighbor":

        gdal.ReprojectImage(
            in_dataset,
            output,
            in_srs.ExportToWkt(),
            in_srs.ExportToWkt(),
            gdalconst.GRA_NearestNeighbour,
        )

    elif interp_method == "Bilinear":
        gdal.ReprojectImage(
            in_dataset,
            output,
            in_srs.ExportToWkt(),
            in_srs.ExportToWkt(),
            gdalconst.GRA_Bilinear,
        )

    elif interp_method == "Average":
        gdal.ReprojectImage(
            in_dataset,
            output,
            in_srs.ExportToWkt(),
            in_srs.ExportToWkt(),
            gdalconst.GRA_Average,
        )

    elif interp_method == "Cubic":
        gdal.ReprojectImage(
            in_dataset,
            output,
            in_srs.ExportToWkt(),
            in_srs.ExportToWkt(),
            gdalconst.GRA_Cubic,
        )
    else:
        raise ValueError("Interpolation method unknown!")

    # Read the reinterpolated raster
    outraster = output.GetRasterBand(1).ReadAsArray()

    # Clean up
    in_dataset = None
    output = None

    return outraster


def resample_dataset(input_ds, input_geotrans, output_ds,
                     output_geotrans, ref_raster, interp_method,
                     epsg=2154, exclude=[], add_padding=False):
    """Resample a gdal dataset.

    Descritption goes here.

    :param input_ds:
    :type input_ds:
    :param input_geotrans:
    :type input_geotrans:
    :param output_ds:
    :type output_ds:
    :param output_geotrans:
    :type output_geotrans:
    :param ref_raster:
    :type ref_raster:
    :param interp_method:
    :type interp_method:
    :param epsg:
    :type epsg: int, optional
    :param exclude:
    :type exclude: list, optional
    :param add_padding:
    :type add_padding: bool, optional
    """
    for array in (x for x in input_ds if x not in exclude):
        arr = resample_raster(input_ds[array].data,
                              input_geotrans,
                              ref_raster,
                              output_geotrans,
                              interp_method, epsg=epsg)
        if add_padding:
            arr = pad(arr[1:-1, 1:-1], ((1, 1), (1, 1)),
                      mode="constant", constant_values=nan)
        output_ds[array] = xr.DataArray(arr, dims=['x', 'y'])


def write_xarray_dset(dataset, outpath, epsg, geotransform):
    """Write an xarray to a geotiff.

    Description goes here.

    :param dataset:
    :type dataset:
    :param outpath:
    :type outpath: str
    :param epsg:
    :type epsg: int
    :param geotransform:
    :type geotransform:
    """
    # Check if all arrays in dataset have the same dimensions
    xsizes = []
    ysizes = []
    # Make a list of all x and y dimensions
    for array in dataset:
        xsizes.append(dataset[array].data.shape[0])
        ysizes.append(dataset[array].data.shape[1])
    # Check if all elements in the list are equal
    if (not xsizes.count(xsizes[0]) == len(xsizes) or not
            ysizes.count(ysizes[0]) == len(ysizes)):
        raise ValueError("Arrays in Dataset aren't the same size!")

    # Define output projection info:
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    srs = srs.ExportToWkt()

    # Create the output image:
    driver = gdal.GetDriverByName('GTiff')
    raster = driver.Create(outpath, ysizes[0], xsizes[0], len(dataset),
                           gdal.GDT_Float32)
    raster.SetProjection(srs)
    raster.SetGeoTransform(geotransform)

    # Iterate over each band
    bandnum = 1
    for array in dataset:
        rasterband = raster.GetRasterBand(bandnum)
        if bandnum == 1:
            rasterband.SetNoDataValue(nan)
        rasterband.SetDescription(array)
        rasterband.WriteArray(dataset[array].data)
        bandnum += 1
        rasterband = None

    # Close the output image
    raster = None
