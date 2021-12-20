#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the complex_terrain algorithm
M. Lamare, M. Dumont, G. Picard (IGE, CEN).
"""
import numpy as np
from numba import njit
from scipy.ndimage import morphology


@njit
def horneslope(indem, dem_pixel_size):
    """ Calculate slope and aspect from a DEM.

    Using jit speeds up the function by a factor of 200.
    The algorithm is based on Horn's (1981) Finite Difference Weighted by
     reciprocal of Distance (Lee and Clarke, 2005). The algorithm has been
     shown to perform better than others (Lee and Clarke, 2005; Srinivasan
     and Engel, 1991; Dunn and Hickley, 1998). The Horn algorithm leads to
     a loss in local variability (smooths small dips and peaks), but doesn't
     overestimate the slopes or exagerate peaks. Jones (1998) showed that
     the algorithm performed well in comparison to reference surfaces.

    Args:
        indem (ndarray): DEM array
        pixel_size (int, int): Pixel size (x and y directions) of the DEM array
            in meters.
    Returns:
        :rtype: (ndarray, ndarray): a tuple containing a ndarray with slope
            values in radians and a ndarray with aspect values in radians."""

    # Initialise the slope and aspect arrays
    output_sl = np.full_like(indem, np.nan)
    output_asp = np.full_like(indem, np.nan)

    rows, cols = indem.shape

    for x in range(1, cols - 1):
        for y in range(1, rows - 1):
            dzx = ((indem[y - 1, x + 1] + 2 * indem[y, x + 1] +
                    indem[y + 1, x + 1]) -
                   (indem[y - 1, x - 1] + 2 *
                    indem[y, x - 1] + indem[y + 1, x - 1])) /\
                         (8 * abs(dem_pixel_size[0]))
            dzy = ((indem[y + 1, x - 1] + 2 * indem[y + 1, x] +
                    indem[y + 1, x + 1]) -
                   (indem[y - 1, x - 1] + 2 * indem[y - 1, x] +
                    indem[y - 1, x + 1])) / (8 * abs(dem_pixel_size[0]))

            slope = np.sqrt(dzx ** 2 + dzy ** 2)
            aspect = 180 / np.pi * np.arctan2(dzy, -dzx)

            output_sl[y, x] = np.arctan(slope) * 180 / np.pi

            if output_sl[y, x] == 0:
                # If no slope, set aspect to 0. Setting it to nan messes up the
                # reinterpolation later.
                output_asp[y, x] = 0
            else:
                if aspect > 90:
                    output_asp[y, x] = 360 - aspect + 90
                else:
                    output_asp[y, x] = 90 - aspect

    return np.deg2rad(output_sl), np.deg2rad(output_asp)


def skyview(horz_data, slp, asp):
    """ Compute the sky view factor.

    Compute the sky view factor for every pixel, i.e. the ratio between diffuse
    irradiance received to that on unobstructed surface according to its slope,
    aspect and horizon. Algorithm by Dozier et al. 1981, 1990.

    Args:
        horz_data (ndarray): horizon elevation data computed from the DEM
        slp (ndarray): slope calculated from the DEM
        asp (ndarray): aspect calculated from the DEM

    Returns:
        vd (ndarray): skyview factor
        ct (ndarray): terrain configuration factor (counterpart of vt)
    """

    Xsize = horz_data.shape[1]  # get the horizon raster size
    Ysize = horz_data.shape[2]
    N = horz_data.shape[0]  # get the number of viewing azimuths from horizon

    # Preallocate variables
    dphi = 2 * np.pi / N
    cosS = np.cos(np.deg2rad(slp))
    sinS = np.sin(np.deg2rad(slp))

    horz_data = np.deg2rad(90 - horz_data)
    cosH = np.cos(horz_data)
    sinH = np.sin(horz_data)

    # Compute vd (skyview factor)
    vd = np.zeros((Xsize, Ysize))  # Initialise vd

    phi = np.degrees(
        np.multiply.outer(
            np.pi - np.array(range(N)) * dphi, np.ones((Xsize, Ysize))
        )
    )

    cos_diff_phi_az = np.cos(np.deg2rad(phi - asp))

    prod = cos_diff_phi_az * (horz_data - sinH * cosH)
    prod[np.isnan(prod)] = 0

    vd_tmp = (dphi / (2 * np.pi)) * (cosS * sinH ** 2 + sinS * prod)
    vd = np.sum(vd_tmp, axis=0)

    ct = 1 - vd

    return vd, ct


def shadows(horz_data, slp, asp, sza, eff_sza, saa):
    """Calculate self, cast and total shadows from a DEM.

    Args:
        horz_data (ndarray): horizon elevation data computed from the DEM
        slp (ndarray): slope calculated from the DEM
        asp (ndarray): aspect calculated from the DEM
        sza (ndarray): solar zenith angles gridded as horz_data
        eff_sza (ndarray): effective solar zenith angles gridded as horz_data
        saa (ndarray): solar azimuth angles gridded as horz_data

    Returns:
        b (ndarray): combined cast and self shadow map. binary product.
            (1 = shadow, 0 = no shadow).
        bs (ndarray): self shadows. Binary product.
        bc (ndarray): cast shadows. binary product."""

    # get number of horizon directions from the horizon file
    N = horz_data.shape[0]

    # Switch horizon data elevation to angle
    horz_data = 90 - horz_data

    # Calculate self shadows (cos gamma is negative)
    bs = np.ones(shape=np.shape(eff_sza))

    # get around error due to Nans
    eff_sza_nonan = np.copy(eff_sza)
    eff_sza_nonan[np.isnan(eff_sza_nonan)] = 1

    # Self-shadows with relaxed (slightly positive angle) value
    bs[np.cos(eff_sza_nonan) < 0.035] = 0

    # Elementary angle between horizon lines
    dphi = 2 * np.pi / N

    # Find the horizon line surrounding a given azimuth
    nn1 = np.int8(np.floor(saa / dphi))
    nn2 = np.int8(np.ceil(saa / dphi))
    m1 = np.uint32(np.mod(N / 2 - nn1, N) + 1)
    m2 = np.uint32(np.mod(N / 2 - nn2, N) + 1)
    m1prodshape = (np.shape(m1)[0] * np.shape(m1)[1])
    m1L = m1prodshape * (m1.flatten() - 1) + np.uint32(
        np.arange(1, m1prodshape + 1, 1)
    )
    m2prodshape = (np.shape(m2)[0] * np.shape(m2)[1])
    m2L = m2prodshape * (m2.flatten() - 1) + np.uint32(
        np.arange(1, m2prodshape + 1, 1)
    )

    # Calculation broken up for clarity
    H1 = np.reshape(horz_data.flatten()[m1L - 1], np.shape(m1))
    H2 = np.reshape(horz_data.flatten()[m2L - 1], np.shape(m2))
    H = np.minimum(H1, H2)

    # Calculate cast shadows
    # In ModImLam the original strict formulatuion:
    # "bc[H < solar_zen] = 1"
    # was relaxed to compute a bit larger, following Sirguey et al. 2009
    # but it overestimates the cast shadows for the Alps
    bc = np.ones(shape=np.shape(H))  # Initialise
    sza_deg = np.rad2deg(sza)
    bc[H < sza_deg] = 0
    #    bc[H < sza_deg + (-0.406 * sza_deg)] = 1

    # Use a morphological operation (erode) to clean shadow mask by removing
    # scattered pixels
    bc_fat = morphology.grey_dilation(bc, size=(3, 3))
    bc = morphology.grey_erosion(bc_fat, size=(3, 3))

    # Calculate the combination of cast and self as binary product
    b = np.logical_and(bs, bc).astype(int)

    return (b, bs, bc)


def effective_zenith_angle(zenith, azimuth, slp, asp):
    """Compute the effective zenith angle.

    The function computes the effective zenith angle given a solar position
     determined by a zenith and azimuth angle, and the slope and aspect of the
     surface.

     Args:
        zenith (int, float, ndarray): solar zenith angle
        azimuth (int, float, ndarray): solar azimuth angle
        slp (int, float, ndarray): slope
        asp (int, float, ndarray): aspect

    Returns:
        (int, float, ndarray): effective solar zenith angle."""

    mu = np.cos(zenith) * np.cos(slp) + np.sin(zenith) * np.sin(slp) *\
        np.cos(azimuth - asp)

    return np.arccos(mu)

def effective_viewing_angles(SZA, SAA, VZA, VAA, slope, aspect):
    """compute the effective viewing angle of incident and observer  as well as the relative azimuth 
    angle between incident and observer for a given slope according to dumont et al.2011
    
    :param SZA: Solar zenith angle (radians)
    :param SAA: Solar azimuth angle (radians)
    :param VZA: viewing zenith angle (radians)
    :param VAA: viewing azimuth angle (radians)
    :param slope: Slope inclination (radians)
    :param aspect: Slope aspect (radians)    
    """
    # Local incident zenith angle
    mu_i = np.cos(SZA) * np.cos(slope) + np.sin(SZA) * \
        np.sin(slope) * np.cos(SAA - aspect)
    # Local viewing zenith angle
    mu_v = np.cos(VZA) * np.cos(slope) + np.sin(VZA) * \
        np.sin(slope) * np.cos(VAA - aspect)

    SZA_eff = np.arccos(mu_i)
    VZA_eff = np.arccos(mu_v)
    # Remove part of the polar representation that correspond to an observer behind the slope, probably already done in REDRESS
    # Local relative azimuth angle (dumont et al.2011)
    mu_az_numerator = (np.cos(VZA) * np.cos(SZA) +
                       np.sin(VZA) * np.sin(SZA) * np.cos(VAA-SAA)
                       - mu_i * mu_v)
    mu_az_denominator = np.sin(SZA_eff) * np.sqrt(1-mu_v**2)
    # When illumination or observator is at nadir (in the new referential), set RAA to zero
    mu_az = np.where(mu_az_denominator != 0, np.divide(
        mu_az_numerator, mu_az_denominator), 0)

    # Garde fou qui prévient d'instabilités numérique autour de -1 et 1
    np.clip(mu_az, -1, 1, out=mu_az)
    #Relative Azimuth Angle effectif
    RAA_eff = np.arccos(mu_az)
    return SZA_eff, VZA_eff, RAA_eff
