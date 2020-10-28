#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unittests for the GDAl tools.

This file is part of the REDRESS algorithm
M. Lamare, M. Dumont, G. Picard (IGE, CEN).
"""
import pytest
from geojson import Polygon, Feature, FeatureCollection, dump
from redress.geospatial.gdal_ops import (build_poly_from_coords,
                                         build_poly_from_geojson,
                                         geom_contains)


@pytest.fixture(scope="session")
def write_geojson(tmpdir_factory):
    """ Write a geojson file with predetermined coordinates."""
    # Create a polygon
    poly = Polygon([[(5.51, 44.71), (6.91, 44.71),
                     (6.91, 45.46), (5.51, 45.46)]])
    # Create a feature
    features = []
    features.append(Feature(geometry=poly))

    # Add to collection
    feature_collection = FeatureCollection(features)

    # Write to file
    fn = tmpdir_factory.mktemp("data").join("poly.geojson")
    with open(fn, 'w') as f:
        dump(feature_collection, f)

    return fn


class Test_polygons(object):
    """Test the extent of built polygons and if they overlap."""

    def test_built_poly(self):
        """Test that a polygon is correctly built from coordinates."""
        # Create 4 coordinates that form a rectangle
        coord_box = [(5.51, 44.71), (6.91, 44.71),
                     (6.91, 45.46), (5.51, 45.46)]

        # Build a polygon
        poly = build_poly_from_coords(coord_box)

        # Test if the coords are correctly built
        assert min(poly.GetEnvelope()) == coord_box[0][0]
        assert max(poly.GetEnvelope()) == coord_box[-1][1]

    def test_geojson_poly(self, write_geojson):
        """Test that a polygon is correctly built from a geojson file."""
        # Get polygon from file
        poly = build_poly_from_geojson(str(write_geojson))

        # Create base coordinates
        coord_box = [(5.51, 44.71), (6.91, 44.71),
                     (6.91, 45.46), (5.51, 45.46)]

        # Test if the coords are correctly built
        assert min(poly.GetEnvelope()) == coord_box[0][0]
        assert max(poly.GetEnvelope()) == coord_box[-1][1]

    def test_poly_contains(self):
        """Test if a created polygon is contained with an other"""

        # Creat a first polygon (not a rectangle)
        main_coord_box = [(6.2869, 45.2729), (6.6165, 45.0735),
                          (6.0919, 45.0191)]
        main_poly = build_poly_from_coords(main_coord_box)

        # Create a polygon inside of the main one
        inside_coords = [(6.2855, 45.1316), (6.3871, 45.1316),
                         (6.3871, 45.1810), (6.2855, 45.1810)]
        inside_poly = build_poly_from_coords(inside_coords)

        # Create a polygon that overlaps the main one
        overlap_coords = [(5.9559, 45.0977), (6.0026, 44.9259),
                          (6.6384, 45.0356)]
        overlap_poly = build_poly_from_coords(overlap_coords)

        # Create a polygon outside the main one
        outside_coords = [(6.6439, 45.7119), (6.8829, 45.7119),
                          (6.8829, 45.8708), (6.6439, 45.8708)]
        outside_poly = build_poly_from_coords(outside_coords)

        assert geom_contains(main_poly, inside_poly)
        assert not geom_contains(main_poly, overlap_poly)
        assert not geom_contains(main_poly, outside_poly)
