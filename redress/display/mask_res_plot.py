#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 10:33:03 2021

@author: nheilir
"""
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
from rasterio import Affine as A
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling,calculate_default_transform
from redress.geospatial.gdal_ops import (resample_raster_diffgeo,write_xarray)
from osgeo import gdal
from matplotlib import colors

def PlotDMPix(outfolder, namepixel, osmd, i, pixelx, pixely, reflectance, mask_bands=None):
    fig = plt.figure()
    wvl = osmd.model.meta["wavelengths"]
    if (mask_bands==None):
        wsl=np.array(wvl)
    else:
        wsl = np.array([wvl[i-1] for i in mask_bands])
   
    plt.plot(wsl,osmd.model.direct_model[:,pixely,pixelx]/osmd.model.a[pixely,pixelx],label="direct_model")
    plt.plot(wvl,osmd.model.rtild[:,pixely,pixelx],label="rtild")
    plt.plot(wvl,osmd.model.r[:,pixely,pixelx],label="r")

    plt.plot(wvl,reflectance[:,pixely,pixelx]/100.,label="reflec")
    plt.scatter(wvl,osmd.model.rtild[:,pixely,pixelx],color="red",s=10)
#    plt.scatter(wsl,osmd.model.direct_model[:,pixely,pixelx],color="red",s=10)
    if (mask_bands!=None):
        plt.scatter(np.array([wvl[i-1] for i in mask_bands]),np.array([osmd.model.rtild[:,pixely,pixelx][i-1] for i in mask_bands]),color="green",s=10)
    plt.xticks(wvl,rotation='vertical',fontsize=6)
#    plt.ylim(0.,2.)
    plt.legend()
    plt.title("%s pixel (%s,%s) %s shadow=%s  \n ssa=%s, a=%s, 1/a*rmse=%s"
              %(osmd.sat.date,pixely,pixelx,namepixel,osmd.sat.topo_bands["all_shadows"].values[pixely,pixelx],osmd.model.ssa[pixely,pixelx],
                osmd.model.a[pixely,pixelx],1/osmd.model.a[pixely,pixelx]*osmd.model.rmse[pixely,pixelx]))
    plt.show()    
    
    fig.savefig(outfolder+"%s%s%s%s%sPixel.pdf"%(osmd.sat.date,i,namepixel, pixelx, pixely), dpi=300, facecolor='w', edgecolor='w',
            orientation='landscape', format='pdf',
            transparent=False, bbox_inches='tight', pad_inches=0.1,
            metadata=None)
    plt.close()
    
def PlotDMPixRADIANCE(outfolder, namepixel, osmd, i, pixelx, pixely, mask_bands=None):
    fig = plt.figure()
    wvl = osmd.model.meta["wavelengths"]
    if (mask_bands==None):
        wsl=np.array(wvl)
    else:
        wsl = np.array([wvl[i-1] for i in mask_bands])

    plt.plot(wvl,np.pi*osmd.sat.sat_band_np[:,pixely,pixelx],label="sat bands")
    plt.plot(wvl,np.pi*np.array(osmd.model.synthetic_toa_radiance)[:,pixely,pixelx],label="model.synthetic_toa_radiance")

#    r_current_divisor = np.array(osmd.model.T_dir_up) * np.array(osmd.model.view_ground) * (np.array(osmd.model.EdP)[:,None,None]+ np.array(osmd.model.EhP)[:,None,None])
    plt.plot(wvl,osmd.model.r_current_divisor[:,pixely,pixelx],label="r_current_divisor")
    
#    plt.scatter(wvl,np.array(osmd.model.synthetic_toa_radiance)[:,pixely,pixelx],color="red",s=10)
#    if (mask_bands!=None):
#        plt.scatter(np.array([wvl[i-1] for i in mask_bands]),np.array([osmd.model.synthetic_toa_radiance[:,pixely,pixelx][i-1] for i in mask_bands]),color="red",s=10)
# 
    plt.xticks(wvl,rotation='vertical',fontsize=6)
    plt.legend()
    plt.title("%s  pixel (%s,%s) %s shadow=%s  \n ssa=%s, a=%s, 1/a*rmse=%s"
              %(osmd.sat.date,pixely,pixelx,namepixel,osmd.sat.topo_bands["all_shadows"].values[pixely,pixelx],osmd.model.ssa[pixely,pixelx],
                osmd.model.a[pixely,pixelx],1/osmd.model.a[pixely,pixelx]*osmd.model.rmse[pixely,pixelx]))
    plt.show()    
    
    fig.savefig(outfolder+"%s%s%s%s%sPixelradiance.pdf"%(osmd.sat.date,i,namepixel, pixelx, pixely), dpi=300, facecolor='w', edgecolor='w',
            orientation='landscape', format='pdf',
            transparent=False, bbox_inches='tight', pad_inches=0.1,
            metadata=None)
    plt.close()   
    
def PlotSsaMaskSnow(outfolder, osmd, i):
    fig = plt.figure()
    ax1 = fig.add_subplot(231)
    c1=ax1.imshow(1./osmd.model.a*osmd.model.rmse,vmin=0.,vmax=0.4)
    plt.colorbar(c1,fraction=0.046, pad=0.04)
    plt.title("1./a*rmse")
    
    ax2 = fig.add_subplot(232)
    c2=ax2.imshow(osmd.model.a,vmin=0.,vmax=1.5)
    plt.colorbar(c2,fraction=0.046, pad=0.04)
    plt.title("a")
    
    ax3 = fig.add_subplot(233)
    c3=ax3.imshow(osmd.model.rmse,vmin=0.,vmax=0.75)
    plt.colorbar(c3,fraction=0.046, pad=0.04)
    plt.title("rmse")
    
    
    
    ax4 = fig.add_subplot(236)
    c4=ax4.imshow(osmd.model.snowmask["isSnow"].values)
    plt.colorbar(c4,fraction=0.046, pad=0.04)
    plt.title("mask")
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    
    ax6 = fig.add_subplot(235)
    c6=ax6.imshow(osmd.sat.topo_bands["all_shadows"],vmin=0.1,vmax=0.9)
    plt.colorbar(c6,fraction=0.046, pad=0.04)
    plt.title("all shadow")
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    
    ax5 = fig.add_subplot(234)
    c5=ax5.imshow(osmd.model.ssa,vmax=100)
    plt.colorbar(c5,fraction=0.046, pad=0.04)
    plt.title("ssa %s iter %s"%(osmd.sat.date,i))
    plt.subplots_adjust(hspace=0.6, wspace=0.6)
    

    
    plt.show()    
    
    fig.savefig(outfolder+"%sMaskSnow%s.pdf"%(osmd.sat.date,i), dpi=300, facecolor='w', edgecolor='w',
        orientation='landscape', format='pdf',
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        metadata=None)
    plt.close()

    
def CompareSnowMask(infolder,dateSAT2,outfolder, osmd, lshadow,i):

    sat2_snowmaskFile= infolder+"SAT2/SnowMask/SAT2_SNOW_%s.tif"%(dateSAT2)
    sat2_snowmask= rasterio.open(sat2_snowmaskFile)
        
    # when layering multiple images, the images need to have the same
    # extent.  This does not mean they need to have the same shape, but
    # they both need to render to the same coordinate system determined by
    # xmin, xmax, ymin, ymax.  Note if you use different interpolations
    # for the images their apparent extent could be different due to
    # interpolation edge effect
    

#
#    figure1, axes = plt.subplots(2, constrained_layout=True,figsize=(8,8))
#    axes[0].imshow(sat2_snowmask.read(1), cmap='gray')
#    axes[1].imshow(osmd.model.snowmask["isSnow"].values, cmap='Greys')
#    
#    axes[0].set_title("Sentinel2 %s \n White:cloud, Grey:snow,\n Black:no snow "%(dateSAT2))
#    axes[1].set_title("Redress %s \n  White:no snow, Black:snow"%(osmd.sat.date))
#     
#    plt.subplots_adjust(hspace=0.5, wspace=0.4)
#    
#    plt.show()
#    figure1.savefig(outfolder+"%sCompareMaskSnow%s.pdf"%(osmd.sat.date,dateSAT2), dpi=300, facecolor='w', edgecolor='w',
#        orientation='landscape', format='pdf',
#        transparent=False, bbox_inches='tight', pad_inches=0.1,
#        metadata=None)
#    plt.close()       
    
###################################################################################   #multi layer senT3
    extent =0, sat2_snowmask.read(1).shape[1],  sat2_snowmask.read(1).shape[0],0
    fig = plt.figure()
    from matplotlib import colors
    cmap = colors.ListedColormap([ 'blue','white', "gray"])
#    plt.imshow(sat2_snowmask.read(1), cmap=cmap ,interpolation='nearest',
#                     extent=extent)
#  
#    plt.imshow(np.ma.masked_where(osmd.model.snowmask["isSnow"].values == 1, osmd.model.snowmask["isSnow"].values), 
#        cmap=plt.cm.Reds_r, interpolation='bilinear', extent=extent, alpha =0.7)
    
    plt.imshow(sat2_snowmask.read(1), cmap=cmap)
    
    plt.imshow(np.ma.masked_where(osmd.model.snowmask["isSnow"].values == 1, osmd.model.snowmask["isSnow"].values), 
        cmap=plt.cm.Reds_r,extent=extent,  alpha =0.7)
    if lshadow:
        plt.imshow(np.ma.masked_where(osmd.sat.topo_bands["all_shadows"]>0.9, osmd.sat.topo_bands["all_shadows"]), 
            cmap=plt.cm.Greens_r,extent=extent,  alpha =0.4)
   
    plt.title("Model %s and SAT2 %s"%(osmd.sat.date,dateSAT2))
    plt.show()    
    fig.savefig(outfolder+"%sCompareMaskSnowoverlay%s%s.pdf"%(osmd.sat.date,dateSAT2,i), dpi=300, facecolor='w', edgecolor='w',
        orientation='landscape', format='pdf',transparent=False, bbox_inches='tight', pad_inches=0.1,metadata=None)
    
    plt.close()    
    
    sat2_snowmask.close()
###################################################################################   

    
def ProjectSnowMaskSAT2(infolder,dateSAT2,outfolder, osmd,i):    
    sat2_snowmaskFile = infolder+"SAT2/SnowMask/SAT2_SNOW_%s.tif"%(dateSAT2)
    sat3_snowmaskFile= outfolder+"%s/subpixelsnow%s.tiff"%(osmd.sat.date,i)
################rasterio    
#    srcRst= rasterio.open(sat2_snowmaskFile)
#    dstRst= rasterio.open(sat3_snowmaskFile)
#
#    kwargs = srcRst.meta.copy()
#    kwargs.update({
#            'crs': dstRst.crs,
#            'transform': dstRst.transform,
#            'width': osmd.sat.shape[1],
#            'height': osmd.sat.shape[0]
#        })
#    
#    #open destination raster
#    dstRstm = np.zeros(dstRst.shape, np.uint8)
#        #reproject and save raster band data
#    for i in range(1, srcRst.count + 1):
#        reproject(
#        source=rasterio.band(srcRst, i),
#        destination=dstRstm,
#        #src_transform=srcRst.transform,
#        src_crs=srcRst.crs,
#        #dst_transform=transform,
#        dst_crs=dstRst.crs,
#        resampling=Resampling.average)
#        
#    dstRstm=np.where(dstRstm<55,0,dstRstm)
#    dstRstm=np.where(dstRstm>150,200,dstRstm)
#    dstRstm=np.where(((dstRstm>55) & (dstRstm<150)),100,dstRstm)  
#    c=plt.imshow(dstRstm)
#    plt.colorbar(c)
#    plt.title("rasterio projection")
#    plt.show()
#
#    #close destination raster
#    dstRst.close()
#    srcRst.close()
#################################################################################################
    
    sat3_snowmask = gdal.Open( sat3_snowmaskFile ) 
    sat2_snowmask = gdal.Open( sat2_snowmaskFile )    
    dest=resample_raster_diffgeo(sat2_snowmask,sat3_snowmask, "Average", 2154, 32631)
    write_xarray(dest, outfolder+"%s/subpixelsnowSAT2.tiff"%(osmd.sat.date), 2154, sat3_snowmask.GetGeoTransform())
#    dest.shape
#    osmd.model.a.shape
    dest=np.where(dest<70,0,dest)
    dest=np.where(dest>150,200,dest)
    dest=np.where(((dest>=70) & (dest<=150)),100,dest)  
    line=4
    col=4
    cmap=colors.ListedColormap([ 'blue', "white"])
    ##HIst masque from sat2
    bins=np.linspace(0, 1.5, 80)
    fig = plt.figure(figsize=(8, 6), dpi=80)
    ax1 = fig.add_subplot(line,col,1)
#    ax1.hist(osmd.model.a[~np.isnan(osmd.sat.topo_bands["all_shadows"])],bins=bins, histtype='bar', ec='black')#sen3
    ax1.hist(osmd.model.a[~np.isnan(osmd.model.rmse)],bins=bins, histtype='bar', ec='black')#sen3
    plt.xlim(bins[0],bins[bins.size-1])
    plt.title("a")
    
    ax3 = fig.add_subplot(line,col,2)
    ax3.hist(osmd.model.a[[((~np.isnan(osmd.model.rmse))&(dest==100) & (osmd.sat.topo_bands["all_shadows"]<=0.9))]], bins=bins, histtype='bar', ec='black')
    plt.xlim(bins[0],bins[bins.size-1])
    plt.title("snow in Sat2 \n with shadow")
    
    ax3 = fig.add_subplot(line,col,3)
    ax3.hist(osmd.model.a[((~np.isnan(osmd.model.rmse))&(dest==100) & (osmd.sat.topo_bands["all_shadows"]>0.9))], bins=bins, histtype='bar', ec='black')
    plt.xlim(bins[0],bins[bins.size-1])
    plt.title("snow in Sat2 \n no shadow")
    
    ax3 = fig.add_subplot(line,col,4)
    ax3.hist(osmd.model.a[((~np.isnan(osmd.model.rmse))&(dest!=100) & (~np.isnan(osmd.sat.topo_bands["all_shadows"])))], bins=bins, histtype='bar', ec='black')
    plt.xlim(bins[0],bins[bins.size-1])
    plt.title("a no snow \n  in Sat2")
    
    bins=np.linspace(0.01, 0.25,80)
    ax2 = fig.add_subplot(line,col,5)
    ax2.hist(osmd.model.rmse[~np.isnan(osmd.model.rmse)], bins=bins, histtype='bar', ec='black')
    plt.xlim(bins[0],bins[bins.size-1])
    plt.title("rmse")
    
    ax3 = fig.add_subplot(line,col,6)
    ax3.hist(osmd.model.rmse[[((~np.isnan(osmd.model.rmse))&(dest==100) & (osmd.sat.topo_bands["all_shadows"]<=0.9))]], bins=bins, histtype='bar', ec='black')
    plt.xlim(bins[0],bins[bins.size-1])
    plt.title("snow in Sat2 \n with shadow")
    
    ax3 = fig.add_subplot(line,col,7)
    ax3.hist(osmd.model.rmse[((~np.isnan(osmd.model.rmse))&(dest==100) & (osmd.sat.topo_bands["all_shadows"]>0.9))], bins=bins, histtype='bar', ec='black')
    plt.xlim(bins[0],bins[bins.size-1])
    plt.title("snow in Sat2 \n no shadow")
    
    ax3 = fig.add_subplot(line,col,8)
    ax3.hist(osmd.model.rmse[((~np.isnan(osmd.model.rmse))&(dest!=100) & (~np.isnan(osmd.sat.topo_bands["all_shadows"])))], bins=bins, histtype='bar', ec='black')
    plt.xlim(bins[0],bins[bins.size-1])
    plt.title("rmse no snow \n  in Sat2")


#    ax3 = fig.add_subplot(line,col,11)
#    c=ax3.imshow(sat2_snowmask.GetRasterBand(1).ReadAsArray(),cmap=cmap)
#    plt.colorbar(c,fraction=0.046, pad=0.04)
#    plt.title("original image")
#    
#    ax3 = fig.add_subplot(line,col,13)
#    c=ax3.imshow(dest,cmap=cmap)
#    plt.colorbar(c,fraction=0.046, pad=0.04)
#    plt.title("gdal projection")
#    
#    ax3 = fig.add_subplot(line,col,14)
#    c=ax3.imshow(osmd.model.snowmask["isSnow"].values,cmap=cmap)
#    plt.colorbar(c,fraction=0.046, pad=0.04)
#    plt.title("REDRESS mask")
    
    ax1 = fig.add_subplot(line,col,9)
#    c1=ax1.imshow(np.where(dest!=100 ,np.nan ,osmd.model.a),vmin=0.5,vmax=1.5)#sen3
    c1=ax1.imshow(osmd.model.a,vmin=0.,vmax=1.5)
    plt.colorbar(c1,fraction=0.046, pad=0.04)
    plt.title("a")
    
    ax2 = fig.add_subplot(line,col,10)
#    c2=ax2.imshow(np.where(dest!=100 ,np.nan ,osmd.model.rmse),vmin=0.15,vmax=0.75)#sen3
    c2=ax2.imshow(osmd.model.rmse,vmin=0.,vmax=0.25)
    plt.colorbar(c2,fraction=0.046, pad=0.04)
    plt.title("rmse")    
    
    ax2 = fig.add_subplot(line,col,11)
    plt.imshow(dest,cmap=colors.ListedColormap([ 'blue', "white"]))
    plt.imshow(osmd.model.snowmask["isSnow"].values, 
        cmap=plt.cm.Reds_r,  alpha =0.7)
#    plt.imshow(np.ma.masked_where(osmd.model.snowmask["isSnow"].values == 1, osmd.model.snowmask["isSnow"].values), 
#        cmap=plt.cm.Reds_r,  alpha =0.7)
    plt.imshow(np.ma.masked_where(osmd.sat.topo_bands["all_shadows"]>0.9, osmd.sat.topo_bands["all_shadows"]), 
        cmap=plt.cm.Greens_r,  alpha =0.4)    
    plt.title("%s VS %s"%(osmd.sat.date,dateSAT2))    
    
    ax2 = fig.add_subplot(line,col,12)
#    c2=ax2.imshow(np.where(dest!=100 ,np.nan ,osmd.model.rmse),vmin=0.15,vmax=0.75)#sen3
    c2=ax2.imshow(osmd.model.ssa,vmax=100)
    plt.colorbar(c2,fraction=0.046, pad=0.04)
    plt.title("ssa")    
    
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.tight_layout(pad=0.2)
    plt.show()    
    
    
    
    
    fig.savefig(outfolder+"%sMaskSat2%s.pdf"%(osmd.sat.date,i), dpi=300, facecolor='w', edgecolor='w',
    orientation='landscape', format='pdf',
    transparent=False, bbox_inches='tight', pad_inches=0.1,
    metadata=None)
    plt.close()
    
    
#    
#    fig = plt.figure()
#    ax1 = fig.add_subplot(221)
#    c1=ax1.imshow(np.where(dest!=100 ,np.nan ,osmd.model.a),vmin=0.5,vmax=1.5)
#    plt.colorbar(c1,fraction=0.046, pad=0.04)
#    plt.title("a")
#    
#    ax2 = fig.add_subplot(222)
#    c2=ax2.imshow(np.where(dest!=100 ,np.nan ,osmd.model.rmse),vmin=0.15,vmax=0.75)
#    plt.colorbar(c2,fraction=0.046, pad=0.04)
#    plt.title("rmse")
#    
#    plt.show()    
#    
#    fig.savefig(outfolder+"%sA_RMSE_MaskSAT2.pdf"%(osmd.sat.date), dpi=300, facecolor='w', edgecolor='w',
#        orientation='landscape', format='pdf',
#        transparent=False, bbox_inches='tight', pad_inches=0.1,
#        metadata=None)
#    plt.close()
#    
    

def PlotHistGenerateMask(outfolder, osmd, i):

    bins=np.linspace(0.5, 1.5, 40)
    fig = plt.figure()
    ax1 = fig.add_subplot(231)
    ax1.hist(osmd.model.a[~np.isnan(osmd.sat.topo_bands["all_shadows"])],bins=bins, histtype='bar', ec='black')
    plt.xlim(bins[0],bins[bins.size-1])
    plt.title("a")
    
    
    ax3 = fig.add_subplot(232)
    ax3.hist(osmd.model.a[osmd.sat.topo_bands["all_shadows"]>0.9], bins=bins, histtype='bar', ec='black')
    plt.xlim(bins[0],bins[bins.size-1])
    plt.title("a no shadow")
    
    ax3 = fig.add_subplot(233)
    ax3.hist(osmd.model.a[osmd.sat.topo_bands["all_shadows"]<=0.9], bins=bins, histtype='bar', ec='black')
    plt.xlim(bins[0],bins[bins.size-1])
    plt.title("a shadow")
    
    bins=np.linspace(0.01, 0.75, 40)
    ax2 = fig.add_subplot(234)
    ax2.hist(osmd.model.rmse[~np.isnan(osmd.sat.topo_bands["all_shadows"])], bins=bins, histtype='bar', ec='black')
    plt.xlim(bins[0],bins[bins.size-1])
    plt.title("rmse")
    
    ax3 = fig.add_subplot(235)
    ax3.hist(osmd.model.rmse[osmd.sat.topo_bands["all_shadows"]>0.9], bins=bins, histtype='bar', ec='black')
    plt.xlim(bins[0],bins[bins.size-1])
    plt.title("rmse no shadow")
    
    ax3 = fig.add_subplot(236)
    ax3.hist(osmd.model.rmse[osmd.sat.topo_bands["all_shadows"]<0.9], bins=bins, histtype='bar', ec='black')
    plt.xlim(bins[0],bins[bins.size-1])
    plt.title("rmse shadow")
    
    plt.subplots_adjust(hspace=0.6, wspace=0.6)
    plt.show()    
    
    
    
    
    fig.savefig(outfolder+"%s%sHistERR.pdf"%(osmd.sat.date,i), dpi=300, facecolor='w', edgecolor='w',
    orientation='landscape', format='pdf',
    transparent=False, bbox_inches='tight', pad_inches=0.1,
    metadata=None)
    plt.close()
