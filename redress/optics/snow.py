#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the complex_terrain algorithm
M. Lamare, M. Dumont, G. Picard (IGE, CEN).
"""
import numpy as np
import sys
sys.path.append('/Users/maxim/dev/spectro')
from spectro import snowoptics, refractive_index


def albedo_kokhanovsky(wls, sza, dd, ssa, B=1.6, g=0.845):

    # Convert wavelengths from nanometers to meters
    wls_m = wls * 1e-9
    # White sky and black sky albedo
    alb_diff_flat_val = snowoptics.albedo_KZ04_diff(wls_m, ssa, BC=0, ni=None,
                                                    B=B, g=g)
    alb_dir_flat = snowoptics.albedo_KZ04_dir(wls_m, sza, ssa, BC=0, ni=None,
                                              B=B, g=g)

    # Make the diffuse albedo the same shape as the sza for consistency
    alb_diff_flat = np.full(sza.shape, alb_diff_flat_val)

    # Blue sky albedo
    alb_kok_flat = snowoptics.albedo_KZ04(wls_m, sza, ssa, r_difftot=dd, BC=0,
                                          ni=None, B=1.6, g=0.845)

    return (alb_dir_flat, alb_diff_flat, alb_kok_flat)


def brf_kokhanovsky(theta_i, theta_v, phi, wvl, SSA, b=13, M=0):
    """ kokhanovsky brF function. Described in
        DOI: 10.1109/LgrS.2012.2185775

       INPUTS:
       - theta_i : illumination zenith angle
       - theta_v: viewing zenith angle
       - phi: relative azimuth angle (illumination - viewing)
       - wvl: wavelength in meters
       - SSA:
       - b: 13, as L = 13d in kokhanovsky's paper
       - M: proportional to the mass concentration of pollutants in snow
    """
    wvl_m = wvl * 1e-9

    def brf0(theta_i, theta_v, phi):
        """ Function to calculate the r0 of the brF
            function. See brf function for details.
        """

        theta = np.arccos(
            -np.cos(theta_i)
            * np.cos(theta_v)
            + np.sin(theta_i)
            * np.sin(theta_v)
            * np.cos(phi)
        ) * 180. / np.pi
        phase = 11.1 * np.exp(-0.087 * theta) + 1.1 * np.exp(-0.014 * theta)
        rr = 1.247 + 1.186 * (np.cos(theta_i) + np.cos(theta_v)) + 5.157 * (
            np.cos(theta_i) * np.cos(theta_v)
        ) + phase
        rr = rr / (4 * (np.cos(theta_i) + np.cos(theta_v)))

        return rr

    # r0 in kokhanovsky's paper
    r = brf0(theta_i, theta_v, phi)

    # k0 for theta_v and theta i
    k0v = 3. / 7. * (1. + 2. * np.cos(theta_v))
    k0i = 3. / 7. * (1. + 2. * np.cos(theta_i))

    # get refractive index from ghislain's spectro library
    n, ni = refractive_index.refice2008(wvl_m)

    gamma = 4 * np.pi * (ni + M) / (wvl_m)

    # Alpha = sqrt(gamma * L), with L approximately 13d, where
    # d is the average optical diameter of snow:
    # d = 6 / refract_ind_ice * SSA
    alpha = np.sqrt(gamma * b * 6. / (917 * SSA))

    # r(theta_i, theta_v, phi)
    rr = r * np.exp(-alpha * k0i * k0v / r)

    return rr
