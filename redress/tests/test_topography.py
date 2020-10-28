#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unittests for the topography module
"""
from pytest import approx
import numpy as np
import xarray as xr
from redress.topography.dem_products import horneslope
from redress.inputs.products import Dem


class Test_dem_prods(object):
    """Test the outputs of the DEM products"""

    def test_slope_aspect(self):
        """Test the slope and aspect of a fake DEM"""

        # Build a fake DEM based on a known product (45º slopes East-West)
        pixel_size = 25.0779  # 25.0779 meter resolution
        dem_a = np.tile(np.array([i * pixel_size for i in range(5)]), (5, 1))
        dem_b = np.flip(dem_a)
        dem_all = np.hstack((dem_a, dem_b))

        # Run the horneslope on the fake DEM
        slope, aspect = horneslope(dem_all, (pixel_size, pixel_size))

        assert slope[2, 2] == approx(np.deg2rad(45))
        assert slope[2, -2] == approx(np.deg2rad(45))
        assert aspect[2, 2] == approx(np.deg2rad(270))
        assert aspect[2, -2] == approx(np.deg2rad(90))


class Test_horizons(object):
    """Test the outputs of the horizon product"""

    def build_dem(self, pix_size):
        """ Build a fake DEM to test the horizon product"""

        # Inherit class
        fake_dem = Dem()

        # Set pixel size, square
        fake_dem.pixel_size = (pix_size, pix_size)

        # Set array (make a fake surface) of 100x100 px
        x, y = np.mgrid[0:2 * np.pi:100j, 0:2 * np.pi:100j]
        z = np.cos(y) * x + np.sin(x) * y
        z = z + 100

        fake_dem.band = xr.DataArray(z)

        return fake_dem

    def extract_horizon(self, elevation_array, distance_array, x, y, N):
        """ Extract horizon elevation and distance values for a given number of
        azimuths """

        # Initialise lists
        philst = []
        elelst = []
        distlst = []

        # Range over phi
        for ii in range(1, 2 * N + 1, 1):
            phi = (ii - 1)*np.pi/N
            philst.append(phi)

            ele_dat = elevation_array[ii-1, :, :]
            dist_dat = distance_array[ii-1, :, :]

            elelst.append(ele_dat[y][x])
            distlst.append(dist_dat[y][x])

        philist_reord = []
        for k in philst:
            if float(k) <= np.pi:
                philist_reord.append(np.pi - float(k))
            else:
                philist_reord.append(np.pi - float(k) + 2*np.pi)

        return philist_reord, elelst, distlst

    def one_point_horizon(self, data, xcoord, ycoord, px, N):
        ''' Compute horizon for a single point in a dem.
        Written by Ghislain '''

        # Get altitude of DEM at the Point of Interest
        alt = data[ycoord, xcoord]
        print('Altitude at point location: %s m' % alt)

        height = data - alt

        ygrid, xgrid = np.mgrid[0:data.shape[0], 0:data.shape[1]]

        xgrid -= xcoord
        ygrid -= ycoord

        distance = px * np.sqrt(ygrid**2+xgrid**2)
        azimuth = np.degrees(np.arctan2(xgrid, ygrid))
        elevation = np.degrees(np.arctan(height / distance))

        elevation[ycoord, xcoord] = 0  # To avoid NaN

        az_res = 360 / (N*2)  # For the directions back and forth

        def argmax2d(A):
            return np.unravel_index(A.argmax(), A.shape)

        azs = np.arange(-180, 180, az_res)

        pa = [argmax2d(np.ma.masked_array(elevation, (azimuth < az) | (
            azimuth > az + az_res))) for az in azs]

        azsrad = [np.pi-i*np.pi/180 for i in azs]

        elelist = []
        distlist = []
        for i in list(range(0, len(azs),)):
            elelist.append(elevation[pa[i]])
            distlist.append(distance[pa[i]])

        return azsrad, elelist, distlist

    def test_horizon_points(self):
        """ Run tests for 2 points on a fake DEM and check elevation / distance
         with basic trig. """

        # Set a point to query to the left of the block
        point_a = (40, 50)

        # Build a DEM with 1 meter resolution
        dem = self.build_dem(1)

        # Compute horizon, with 64 directions
        dem.compute_horizon(N=32)

        # Extract horizon data at a given point
        ct_phi, ct_ele, ct_dist = self.extract_horizon(
            dem.topo_bands["horizon_ele"].data,
            dem.topo_bands["horizon_dist"].data, point_a[0], point_a[1], N=32)

        # Extract horizon using the single point algorithm as comparison point
        onept_az, onept_ele, onept_dist = self.one_point_horizon(
            dem.band.data, point_a[0], point_a[1], dem.pixel_size[0], 32)

        # Compare distances to horizon 5% tolerence on mean
        assert np.average(np.array(ct_dist)) == approx(
            np.average(np.array(onept_dist)), 5)
        # Compare values of elevation to horizon 5% tolerence on mean
        assert np.average(np.array(ct_ele)) == approx(
            np.average(np.array(onept_ele)), 4.5)
