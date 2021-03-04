#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Snow optics tools.

This file is part of the complex_terrain algorithm
M. Lamare, M. Dumont, G. Picard (IGE, CEN).
"""
import numpy as np
import sys
sys.path.append('/home/nheilir/REDRESS/snowoptics/')
import snowoptics as so


def albedo_kokhanovsky(wls, sza, dd, ssa, B=1.6, g=0.845):
    """Calculate black sky, white sky and blue sky albedo.

    Using the Asymptotic Radiative Transfer theory from Kokhanovsky and Zege,
    2004 (implemented in the snowoptics library by Ghislain Picard:
     https://github.com/ghislainp/snowoptics), calculate direct (black sky),
    diffuse (white sky) and total (blue sky) albedo for a given snow SSA, SZA,
    and direct-to-diffuse ratio. The calculations rely on the B and g values
    found in Libois et al. 2013 (doi:10.5194/tc-7-1803-2013).

    :param wls: Array of wavelength values at which to compute the albedo (nm)
    :type wls: ndarray
    :param sza: Solar Zenith Angle grid (rad)
    :type sza: ndarray
    :param dd: Direct-to-diffuse ratio values at the given wavelengths
    :type dd: ndarray
    :param ssa: Specific surface area of the snow (m2/kg)
    :type ssa: float
    :param B: Snow absorption enhancement parameter
    :type B: float, optional
    :param g: Snow asymmetry factor
    :type g: float, optional
    :return: black sky, white sky and blue sky albedos
    :rtype: tuple [3]
    """
    # Convert wavelengths from nanometers to meters (done here to avoid
    # confusion).
    wls_m = wls * 1e-9
    # White sky albedo
    alb_diff_flat_val = so.albedo_diffuse_KZ04(wls_m, ssa,
                                               impurities={'BC': 0},
                                               ni="w2008", B=B, g=g)
                                               
    alb_diff_flat_val= np.where(np.isnan(ssa),0.2, alb_diff_flat_val)
    # alb_diff_flat_val= np.where(np.isnan(ssa),0.0, alb_diff_flat_val)
    
    # Black sky albedo
    alb_dir_flat = so.albedo_direct_KZ04(wls_m, sza, ssa,
                                         impurities={'BC': 0},
                                         ni="w2008",
                                         B=B, g=g)
    alb_dir_flat= np.where(np.isnan(ssa),0.2, alb_dir_flat)
    # alb_dir_flat= np.where(np.isnan(ssa),0.0, alb_dir_flat)

    # Make the diffuse albedo the same shape as the sza for consistency
    alb_diff_flat = np.full(sza.shape, alb_diff_flat_val)

    # Blue sky albedo
    alb_kok_flat = so.albedo_KZ04(wls_m, sza, ssa, r_difftot=dd,
                                  impurities={'BC': 0},
                                  ni="w2008", B=1.6, g=0.845)
    alb_kok_flat= np.where(np.isnan(ssa),0.2, alb_kok_flat)  
    # alb_kok_flat= np.where(np.isnan(ssa),0.0, alb_kok_flat)                                  

    return (alb_dir_flat, alb_diff_flat, alb_kok_flat)


def brf_kokhanovsky(theta_i, theta_v, phi, wvl, SSA, b=13, M=0):
    """Calculate snow BRF.

    Snow BRF function from Kokhanovsky et al. Described in DOI:
    10.1109/LgrS.2012.2185775. Calculate BRF value for given illumination /
    viewing angles, wavelengths and SSA.

    :param theta_i: illumination zenith angle
    :type theta_i: float
    :param theta_v: viewing zenith angle
    :type theta_v: float
    :param phi: relative azimuth angle (illumination - viewing)
    :type phi: float
    :param wvl: wavelength (m)
    :type wvl: ndarray
    :param SSA: Specific surface area of snow
    :type SSA: float
    :param b: 13, as L = 13d in kokhanovsky's paper
    :type b: float, optional
    :param M: proportional to the mass concentration of pollutants in snow
    :type M: float, optional
    :return: BRF
    :rtype: ndarray
    """
    wvl_m = wvl * 1e-9

    def brf0(theta_i, theta_v, phi):
        """Calculate the r0 of the BRF.

        See brf function for details.

        :param theta_i: illumination zenith angle
        :type theta_i: float
        :param theta_v: viewing zenith angle
        :type theta_v: float
        :param phi: relative azimuth angle (illumination - viewing)
        :type phi: float
        :return: r0
        :rtype: float
        """
        theta = np.arccos(-np.cos(theta_i) * np.cos(theta_v) + np.sin(theta_i)
                          * np.sin(theta_v) * np.cos(phi)) * 180. / np.pi
        phase = 11.1 * np.exp(-0.087 * theta) + 1.1 * np.exp(-0.014 * theta)
        rr = 1.247 + 1.186 * (np.cos(theta_i) + np.cos(theta_v)) + 5.157 * (
            np.cos(theta_i) * np.cos(theta_v)) + phase
        rr = rr / (4 * (np.cos(theta_i) + np.cos(theta_v)))

        return rr

    # r0 in kokhanovsky's paper
    r = brf0(theta_i, theta_v, phi)

    # k0 for theta_v and theta i
    k0v = 3. / 7. * (1. + 2. * np.cos(theta_v))
    k0i = 3. / 7. * (1. + 2. * np.cos(theta_i))

    # get refractive index
    n, ni = so.refractive_index.refice2008(wvl_m)

    gamma = 4 * np.pi * (ni + M) / (wvl_m)

    # Alpha = sqrt(gamma * L), with L approximately 13d, where
    # d is the average optical diameter of snow:
    # d = 6 / refract_ind_ice * SSA
    alpha = np.sqrt(gamma * b * 6. / (917 * SSA))

    # r(theta_i, theta_v, phi)
    rr = r * np.exp(-alpha * k0i * k0v / r)
    #if ssa is nan (no snow) the brf value must be 0.2
    rr.values= np.where(np.isnan(SSA),0.2, rr.values)
    # rr.values= np.where(np.isnan(SSA),0.0, rr.values)    


    return rr
