# -*- coding: utf-8 -*-

"""Module regroupant des fonctions de traitement générique (en général) d'intéret pour différents instruments et/ou différents sites.

.. warning::
   Les fonctions de ce module modifient le spectre fourni en argument directement (opération 'in-place'), c'est dire que les données (attribut ``data``) sont remplacées par les données traitées. C'est un choix de programmation fait pour des raisons de performances en mémoire et en temps (mais pas vérifié à ce jour) et pour permettre le caching de façon simple, mais n'est pas très élegant et interdit la programmation sous forme de 'pipeline' ou de 'workflow'. Il est donc possible que cela change à l'avenir, ce qui devrait avoir un impact limité sur les codes, mais demandera quand même quelques modifications.

"""

import collections
import six
import numpy as np
import math
import scipy.optimize
import scipy.signal
import sys
sys.path.append('/home/nheilir/REDRESS/snowoptics/')
import snowoptics as so
import xarray as xr


# utility functions to write more condensed processing too.
class AlgorithmError(Exception):
    pass


def ssa_estimate(osmd, mask=None, model="2-param"):
    """estimate the SSA by fitting the spec to the model specified in parameter

    :param spec: albedo spec
    """
    def unpack_params(params):
        if model == "1-param":
            a, b = 1.0, 0.0
        if model == "2-param":
            a, b = params[0], 0.0
        elif model == "3-param":
            a, b = params[0:2]
        return a, b, params[-1]

    def direct_model(params, wls, dirdiff_ratio, sza):
        a, b, ssa = unpack_params(params)
        return a * (1 - b * (wls-400)/(1050-400)) * so.albedo_KZ04(wls * 1e-9,
                                                   sza, ssa, 
                                                   r_difftot=dirdiff_ratio, impurities=None, ni="p2016")

    def cost_difference(params, spec, dirdiff_ratio, sza, wls): 
        return spec - direct_model(params, wls, dirdiff_ratio, sza)

    
    if model == "1-param":
        first_guess = [40.]
        params_guess=np.zeros((osmd.sat.shape[0],osmd.sat.shape[1],1))
    elif model == "2-param":
        first_guess = [1.0, 40.]
        params_guess=np.zeros((osmd.sat.shape[0],osmd.sat.shape[1],2))
    elif model == "3-param":
        first_guess = [1.0, 0.0, 40.]   
        params_guess=np.zeros((osmd.sat.shape[0],osmd.sat.shape[1],3))
    
    dirdiff_ratio=np.array([a/(a+b) for a,b in zip(osmd.model.EdP,osmd.model.EhP)])
    if mask==None :
        wls = np.array(osmd.sat.meta["wavelengths"])
        ddr = np.array(dirdiff_ratio)
        specM = osmd.model.rtild
    else:
        wls = np.array([osmd.sat.meta["wavelengths"][i-1] for i in mask])
        ddr = np.array([dirdiff_ratio[i-1] for i in mask])
        specM = np.array([osmd.model.rtild[i-1] for i in mask])
        
    osmd.model.a=np.zeros(osmd.sat.shape)
    osmd.model.direct_model=np.zeros((len(wls),osmd.sat.shape[0],osmd.sat.shape[1]))
    osmd.model.rmse=np.zeros(osmd.sat.shape)
    
    for y in range(0,osmd.sat.shape[0]):
        for x in range(0,osmd.sat.shape[1]):
            params_guess[y,x,:], it = scipy.optimize.leastsq(cost_difference, first_guess, 
                      args=(specM[:,y,x], ddr[:,y,x], osmd.model.topo_bands["eff_sza"].values[y,x], wls), epsfcn=0.001)

            osmd.model.a[y,x], b, osmd.model.ssa[y,x] = unpack_params(params_guess[y,x,:])
            osmd.model.direct_model[:,y,x] = direct_model(params_guess[y,x,:],wls,
                                                        ddr[:,y,x],osmd.model.topo_bands["eff_sza"].values[y,x])
            osmd.model.rmse[y,x] = math.sqrt(np.mean((specM[:,y,x] - osmd.model.direct_model[:,y,x]) ** 2))
    
    osmd.model.ssa= np.where(osmd.model.ssa>250,250,osmd.model.ssa)


def _alb_kokha(wls, sza, direct_diffus_ratio, ssa):
    n, ni = so.refractive_index.refice2008(wls)
    albedo_dir = kokha_dir(wls, sza, ssa, ni, b=4.3)
    albedo_dif = kokha_diff(wls, ssa, ni, b=4.3)
    albedo_th = (direct_diffus_ratio * albedo_dir + albedo_dif) / (direct_diffus_ratio + 1.0)
    return albedo_th

def kokha_dir(lamb, sza, ssa, ni=None, b=4.53):
    if ni is None:
        n, ni = so.refractive_index.refice2008(lamb)
    rho = 917.0
    return np.exp(-3.0/7.0 * (1+2*np.cos(sza)) * b * np.sqrt(24*np.pi*ni / (lamb*rho*ssa)))


def kokha_diff(lamb, ssa, ni=None, b=4.53):
    if ni is None:
        n, ni = so.refractive_index.refice2008(lamb)
    rho = 917.0
    return np.exp(- b * np.sqrt(24*np.pi*ni / (lamb*rho*ssa)))


def mask_estimate(osmd):
    osmd.model.snowmask["isSnow"]=xr.DataArray(np.where(((1./osmd.model.a*osmd.model.rmse)<0.3) & (osmd.model.a<4),1,np.nan),
                                              dims=['y', 'x'],
                                              coords=osmd.model.topo_bands.coords)

 
def refl_estimate(osmd):
    
    r_current_dividend = np.pi * (np.array(osmd.model.synthetic_toa_radiance) - np.array(osmd.model.LtNA) - np.array(osmd.model.LtA)[:,None,None])
    
    r_current_divisor = np.array(osmd.model.T_dir_up) * np.array(osmd.model.view_ground) * (np.array(osmd.model.EdP)+ np.array(osmd.model.EhP))
    
    osmd.model.r = np.divide(r_current_dividend, r_current_divisor,
                              out=np.zeros_like(r_current_dividend),
                              where=r_current_divisor != 0)

        
def obs(osmd):

    sat_bands=np.zeros((len(osmd.model.meta["wavelengths"]),osmd.sat.shape[0],osmd.sat.shape[1]))
    for band,index in zip(osmd.model.meta["bandlist"],range(len(osmd.model.meta["wavelengths"]))):
        sat_bands[index,:,:]= osmd.sat.bands[band].values
    
    r_current_dividend = np.pi * (sat_bands - np.array(osmd.model.LtNA) - np.array(osmd.model.LtA)[:,None,None])
    
    r_current_divisor = np.array(osmd.model.T_dir_up) * np.array(osmd.model.view_ground) * (np.array(osmd.model.EdP) + np.array(osmd.model.EhP))
     
    
    osmd.model.rtild = np.divide(r_current_dividend, r_current_divisor,
                              out=np.zeros_like(r_current_dividend),
                              where=r_current_divisor != 0)

        
