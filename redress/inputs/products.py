#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is part of the complex_terrain algorithm
M. Lamare, M. Dumont, G. Picard (IGE, CEN).
"""
import importlib
import xarray as xr
import numpy as np
from scipy.ndimage import median_filter
from redress.topography.horizon import dozier_horizon
from redress.topography import dem_products
from redress.optics.snow import *
from redress.topography.forward_model import Cases, iterative_radiance
import redress.optics.snow_mask as sm
from pathlib import Path
from redress.inputs.geotiffs import import_DEM_gtiff
from redress.geospatial.gdal_ops import (build_poly_from_geojson,
                                                resample_dataset,
                                                write_xarray_dset)
from redress.inputs.satellites import import_s3OLCI

class Sat(object):
    """ Satellite image class

    This class is composed of the necessary bands and metadata to run the
    forward model. To populate this class, a specific reader has to be
    specified in order to import the satellite image.
     """
    def __init__(self):
        
        self.date = " "
        self.meta = {"path": None, "bandlist": [], "angles": [],
                     "wavelengths": []}
        self.bands = xr.Dataset()
        self.altitude = xr.Dataset()
        self.geotransform = None
        self.pixel_size = None
        self.shape = None
        self.angles = xr.Dataset()
        self.topo_bands = xr.Dataset()
        self.snowmask =  xr.Dataset()

class Dem(object):
    """ DEM image class

    This class stores the DEM altitude band, the geocoding information and the
    resampled solar / viewing angles provided.
    """
    def __init__(self):
        self.meta = {"path": None, "bandlist": [],}
        self.bands = xr.Dataset()
        self.geotransform = None
        self.pixel_size = None
        self.angles = xr.Dataset()
        self.geo_extent=None

    def compute_horizon(self, N=32):
        """Compute horizon elevation and distance.

        The algorithm, based on Dozier et al. (1981) calculates the elevation
         and distance to the horizon for each pixel in a DEM, for a number of
        directions. Sirguey et al., 2009 showed although Dozier et al. 1981
        suggested that 32 directions were sufficient to obtain a decent skyview
        factor, 64 directions help obtain a more accurate representation of
        shadows. Here N results in 2N directions (N32 = 64 directions).

        Args:
            dem (self): dem object
            N (int): Number of directions to be calculated divided by 2.

        """

        # Test if the pixels are square
        if not abs(self.pixel_size[0]) == abs(self.pixel_size[1]):
            raise RuntimeError("Error: Horizon algorithm only works with"
                               " square pixels!")

        # Set nans to zeros for the horizon to work (zero is impossible in the
        # mountains)
        self.bands["altitude"].fillna(0)

        # Pre-allocate elevation and distance rasters. These rasters will
        # contain the angle in degrees to the horizon and the distance in
        # meters for each pixel. Each slice in the 3rd dimension represents
        # a direction.
        eh = np.full((2 * N, self.bands["altitude"].data.shape[0],
                      self.bands["altitude"].data.shape[1]), np.NaN,
                     dtype=np.float32,
                     )
        dh = np.full((2 * N, self.bands["altitude"].data.shape[0],
                      self.bands["altitude"].data.shape[1]), np.NaN,
                     dtype=np.float32,
                     )

        # Run the horizon algorithm for each direction
        for ii in range(1, N + 1, 1):
            phi = (ii - 1) * np.pi / N
            ehtemp, dhtemp = dozier_horizon(self.bands["altitude"].data,
                                            abs(self.pixel_size[0]), phi)
            dh[ii - 1, :, :] = dhtemp[0, :, :]
            eh[ii - 1, :, :] = ehtemp[0, :, :]
            dh[ii - 1 + N, :, :] = dhtemp[1, :, :]
            eh[ii - 1 + N, :, :] = ehtemp[1, :, :]

        # Remove the noise in the products by applying a light median filter
        eh = median_filter(eh, 3)
        dh = median_filter(dh, 3)

        # Convert numpy arrays to xarray
        self.bands["horizon_ele"] = xr.DataArray(eh,
                                                 dims=['phi', 'y', 'x'])
        self.bands["horizon_dist"] = xr.DataArray(dh,
                                                  dims=['phi', 'y', 'x'])

        # Set zeros back to nans
        self.bands["altitude"].where(self.bands["altitude"] == 0)

    def compute_topo_bands(self):
        """Calculate all topographic products from DEM

        Calculates all the topographic products based on the input DEM and the
        derived products.

        Args:
            self """

        # Temporarily store products in a dictionnary
        topo_prods = {}

        # Check if horizon exists
        if "horizon_ele" not in self.bands.keys():
            raise ValueError("Please calculate or load the horizon product"
                             " before computing other topographic products!")

        # Compute aspect and slope of DEM
        print("Calculating Slope and Aspect from DEM\n")
        topo_prods["slope"], topo_prods["aspect"] = dem_products.horneslope(
            self.bands["altitude"].data, self.pixel_size)

        # Calculate effective sza
        print("Calculating Effective Solar and Viewing Angles\n")
        topo_prods["eff_sza"] = dem_products.effective_zenith_angle(
            self.angles["SZA"],
            self.angles["SAA"],
            topo_prods["slope"], topo_prods["aspect"])

        # Calculate effective oza
        topo_prods["eff_vza"] = dem_products.effective_zenith_angle(
            self.angles["VZA"], self.angles["VAA"],
            topo_prods["slope"], topo_prods["aspect"])

        # Compute skyview and terrain configuration factors
        print("Calculating Skyview Factor\n")
        topo_prods["vt"], topo_prods["ct"] = dem_products.skyview(
            self.bands["horizon_ele"].data,
            topo_prods["slope"], topo_prods["aspect"])

        # Calculate shadow product
        print("Calculating pixels based shadows\n")
        topo_prods["all_shadows"], topo_prods["self_shadows"],\
            topo_prods["cast_shadows"] = dem_products.shadows(
                self.bands["horizon_ele"].data,
                topo_prods["slope"], topo_prods["aspect"],
                self.angles["SZA"].data,
                topo_prods["eff_sza"],
                self.angles["SAA"].data)

        # Pad the bands with zeros around the edges to remove the aberrant
        # values, then save the array to the Dataset
        for key, value in topo_prods.items():
            padded_value = np.pad(value[1:-1, 1:-1], ((1, 1), (1, 1)),
                                  mode="constant")
            self.bands[key] = xr.DataArray(padded_value,
                                           dims=['y', 'x'],
                                           coords=self.bands.coords)

        print("Done\n")


class Model(object):
    """ Model output class

    This class is composed of the necessary bands and metadata to run the
    forward model. To populate this class, a specific reader has to be
    specified in order to import the satellite image.
     """
    def __init__(self):

        self.ssa = None
        self.angles = None
        self.dem = None
        self.meta = None
        self.rt_model = None
        self.rt_options = {}
        self.difftot = []
        self.brf = []
        self.albedo = {"total": [], "direct": [], "diffuse": []}
        self.case = Cases()
        self.toa_rad = xr.Dataset()
        self.snowmask =  xr.Dataset()
        self.EdP=[]
        self.EhP=[]
        self.synthetic_toa_radiance=[]
        self.LtNA=[]
        self.LtA =[]
        self.T_dir_up=[]
        self.view_ground=[]
        self.rtild = []
        self.r = []
        self.a = []
        self.rmse = []
        self.direct_model = []
        self.geotransform = None
        
        
    # TODO: make a copy attributes function in class to copy attributes from other class
    def import_rt_model(self, rt_name):
        """ Import a model"""
        # Try to import the model in the rt_models folder.
        # Return the reader class in the module.
        try:
            module = importlib.import_module(
                "redress.rtmodel.%s" % rt_name)
        except ImportError as e:
            raise IOError("Unable to find the rt model %s. %s." % (rt_name, e))

        self.rt_model = getattr(module, "rtmodel")

    def set_rt_options(self, aero, aod=0.1, refl=0.99, water=0.05,
                       ozone=0.3, atmos=None, atcor=False):
        """Set the atmopheric conditions as entry options for the RT Model."""
        self.rt_options.update({"aerosol_model": aero,
                                "aod": aod, "refl": refl, "water": water,
                                "ozone": ozone, "atmo_model": atmos,
                                "atcor": atcor})

    def compute_albedo(self):
        if not self.ssa.all():
            raise ValueError("The ssa of snow hasn't been set!")

        if not self.rt_model:
            raise ValueError("No RT model specified!")

        if not self.rt_options:
            raise ValueError("The input options to run the RT model haven't"
                             " been set.")

        # Calculate dirdiff for the moment calculate for the mean over the
        #  scene
        for wvl in self.meta["wavelengths"]:
            rt = self.rt_model()
            rt.run(self.angles["SZA"].data.mean(),
                   self.angles["SAA"].data.mean(),
                   self.angles["VZA"].data.mean(),
                   self.angles["VAA"].data.mean(),
                   wvl,
                   np.nanmean(self.topo_bands["altitude"].data),
                   self.rt_options["aerosol_model"],
                   aod=self.rt_options["aod"],
                   refl=self.rt_options["refl"],
                   water=self.rt_options["water"],
                   ozone=self.rt_options["ozone"],
                   atmo=self.rt_options["atmo_model"],
                   atcor=self.rt_options["atcor"],
                   )
            dir_illum = rt.outputs.direct_solar_irradiance
            diff_illum = rt.outputs.diffuse_solar_irradiance

            diffuse_total_ratio = 1 / (1 + dir_illum / diff_illum)

            self.difftot.append(diffuse_total_ratio)

            # Compute albedo
            alb_dir, alb_diff, alb_tot = albedo_kokhanovsky(
                wvl,
                self.angles["SZA"].data,
                diffuse_total_ratio,
                self.ssa,
                B=1.6, g=0.845)
            self.albedo["direct"].append(alb_dir)
            self.albedo["diffuse"].append(alb_diff)
            self.albedo["total"].append(alb_tot)

    def compute_brf(self):
        # Calculate brf factor for geometry, wavelength and ssa value
        for wvl in self.meta["wavelengths"]:
            self.brf.append((wvl,
                             brf_kokhanovsky(self.topo_bands["eff_sza"],
                                                  self.topo_bands["eff_vza"],
                                                  self.angles["SAA"] -
                                                  self.angles["VAA"],
                                                  wvl, self.ssa,)))

    def simulate_TOA_radiance(self, casenum, osmd):

        # Compute albedo
        self.compute_albedo()

        # Compute BRF
        self.compute_brf()

        case = Cases()
        case.create_cases(casenum)
        self.EdP=[]
        self.EhP=[]
        self.synthetic_toa_radiance=[]
        self.LtNA=[]
        self.LtA =[]
        self.T_dir_up=[]
        self.view_ground=[]
        for band, wvl, hdr, bhr, brf in zip(self.meta["bandlist"],
                                            self.meta["wavelengths"],
                                            self.albedo["direct"],
                                            self.albedo["diffuse"],
                                            self.brf,
                                            ):

            synthetic_toa_radiance, LtNA, LtA, T_dir_up, view_ground, EdP, EhP,E0eff = iterative_radiance(
                                            self.topo_bands,
                                            self.angles,
                                            wvl,
                                            hdr,
                                            bhr,
                                            self.rt_model(),
                                            self.rt_options,
                                            brf[1].data,
                                            case,                                                        
                                            tw=5,
                                            aw=7,
                                            dif_anis=False,
                                            )
            self.toa_rad[band] = xr.DataArray(synthetic_toa_radiance,
                                              dims=['y', 'x'],
                                              coords=self.topo_bands.coords)

            self.EdP.append(EdP)
            self.EhP.append(EhP)          
            self.synthetic_toa_radiance.append(synthetic_toa_radiance)
            self.LtNA.append(LtNA)
            self.LtA.append(LtA)
            self.T_dir_up.append(T_dir_up)
            self.view_ground.append(view_ground)
            
        
        
class SMD (object):
    """ SMD class

    This class is composed of the satellite image,dem and model.
     """
    def __init__(self):

        self.sat = None
        self.dem = None
        self.model = None
                        
    def init_DEM(self, outfolder,infolder, strdem, strpoly ,strdemread):
        """ initialise the objet dem in  SMD """
        
        # DEM path
        dem_path = Path(infolder+strdem)
        
        # Provide a geojson file and build polygon
        bbox_json = Path(infolder+strpoly)
        geo_extent = build_poly_from_geojson(str(bbox_json))
        
        # import data
        self.dem = import_DEM_gtiff(dem_path, geo_extent, epsg=2154, reader=strdemread)
        write_xarray_dset(self.dem.bands, outfolder+strdem, 2154, self.dem.geotransform, ignore=[])
        
        self.dem.geo_extent=geo_extent # AVOIR
        
    def init_SAT(self, outfolder,infolder, date, strsat,strsatread):
        """ initialise the objet sat in  SMD """
        s3_path = Path(strsat)
        self.sat = import_s3OLCI(s3_path, self.dem.geo_extent, epsg=2154, reader=strsatread)
        self.sat.date = date 
        write_xarray_dset(self.sat.bands, outfolder+"%s/sen.tif" %date, 2154, self.sat.geotransform, ignore=[])


        # Resample datasets
        resample_dataset(self.sat.angles, self.sat.geotransform, self.dem.angles, self.dem.geotransform,
                         self.dem.bands["altitude"], "NearestNeighbor")
    
        write_xarray_dset(self.dem.angles, outfolder+"%s/dem2.tif" %date, 2154, self.dem.geotransform, ignore=[])
        self.sat.shape=(self.sat.bands.sizes["y"],self.sat.bands.sizes["x"])
        #################
        # EITHER
        # Open horizon file
        saved_horizon = xr.open_dataset(infolder+"horizons_large.nc")
        # Update the dem product
        self.dem.bands.update(saved_horizon)
        #################
        # # OR
        # # Calulate the horizon
        # self.dem.compute_horizon()
        
        # # To save the horizon arrays (elevation and distance), merge them to 1 Dataset
        # horizons = xr.merge([self.dem.bands["horizon_ele"], self.dem.bands["horizon_dist"]])
        # #print(horizons)
        
        # # Save horizon file to a netcdf
        # hor_file = Path("/home/lamarem/Documents/REDRESS/outputs/horizons.nc")
        # hor_file ="horizons.nc"
        # horizons.to_netcdf(hor_file)
        
        #  self.dem.bands.update(horizons)
        
        # # Close the dataset to free memory
        # horizons = None
        ##################
        
        # Compute other topobands
        self.dem.compute_topo_bands()
        
        write_xarray_dset(self.dem.bands,outfolder+"dem_topo.tif", 2154, self.dem.geotransform, ignore=["horizon_ele", "horizon_dist"])
    
        # Resample to satellite
        resample_dataset(self.dem.bands, self.dem.geotransform, self.sat.topo_bands,
                         self.sat.geotransform, self.sat.bands["Oa01"],
                         "Average",
                         epsg=2154, exclude=["horizon_ele", "horizon_dist"],
                         add_padding=True)
    
    
        write_xarray_dset(self.sat.topo_bands, outfolder+"%s/topobands.tiff"% date, 2154, self.sat.geotransform, ignore=[])
        write_xarray_dset(self.sat.bands,
                          outfolder+"%s/sat_bands.tiff" % date, 2154,
                          self.sat.geotransform)
        write_xarray_dset(self.sat.angles,
                          outfolder+"%s/angles.tiff" % date, 2154,
                          self.sat.geotransform)
        
