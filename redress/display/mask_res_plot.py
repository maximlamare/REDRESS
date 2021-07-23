#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 10:33:03 2021

@author: nheilir
"""
import matplotlib.pyplot as plt
import numpy as np

def PlotDMPix(outfolder, namepixel, osmd, i, pixelx, pixely, mask_bands=None):
    fig = plt.figure()
    wvl = osmd.model.meta["wavelengths"]
    if (mask_bands==None):
        wsl=np.array(wvl)
    else:
        wsl = np.array([wvl[i-1] for i in mask_bands])
   
    plt.plot(wsl,osmd.model.direct_model[:,pixely,pixelx]/0.9775,label="direct_model")
    plt.plot(wvl,osmd.model.rtild[:,pixely,pixelx],label="rtild")
    plt.plot(wvl,osmd.model.r[:,pixely,pixelx],label="r")
    plt.scatter(wvl,osmd.model.rtild[:,pixely,pixelx],color="green",s=10)
#    plt.scatter(wsl,osmd.model.direct_model[:,pixely,pixelx],color="red",s=10)
    if (mask_bands!=None):
        plt.scatter(np.array([wvl[i-1] for i in mask_bands]),np.array([osmd.model.rtild[:,pixely,pixelx][i-1] for i in mask_bands]),color="red",s=10)
    plt.xticks(wvl,rotation='vertical',fontsize=6)
    plt.legend()
    plt.title("%s iteration %s pixel %s (%s,%s) \n ssa=%s, a=%s, 1/a*rmse=%s"
              %(osmd.sat.date,i,namepixel,pixely,pixelx,osmd.model.ssa[pixely,pixelx],
                osmd.model.a[pixely,pixelx],1/osmd.model.a[pixely,pixelx]*osmd.model.rmse[pixely,pixelx]))
    plt.show()    
    
    fig.savefig(outfolder+"%s%s%sPixel.pdf"%(osmd.sat.date,i,namepixel), dpi=300, facecolor='w', edgecolor='w',
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

    r_current_divisor = np.array(osmd.model.T_dir_up) * np.array(osmd.model.view_ground) * (np.array(osmd.model.EdP)+ np.array(osmd.model.EhP))
    plt.plot(wvl,r_current_divisor[:,pixely,pixelx],label="r_current_divisor")
    
#    plt.scatter(wvl,np.array(osmd.model.synthetic_toa_radiance)[:,pixely,pixelx],color="red",s=10)
#    if (mask_bands!=None):
#        plt.scatter(np.array([wvl[i-1] for i in mask_bands]),np.array([osmd.model.synthetic_toa_radiance[:,pixely,pixelx][i-1] for i in mask_bands]),color="red",s=10)
# 
    plt.xticks(wvl,rotation='vertical',fontsize=6)
    plt.legend()
    plt.title("%s iteration  %s pixel %s (%s,%s) \n ssa=%s, a=%s, 1/a*rmse=%s"
              %(osmd.sat.date,i,namepixel,pixely,pixelx,osmd.model.ssa[pixely,pixelx],
                osmd.model.a[pixely,pixelx],1/osmd.model.a[pixely,pixelx]*osmd.model.rmse[pixely,pixelx]))
    plt.show()    
    
    fig.savefig(outfolder+"%s%s%sPixelradiance.pdf"%(osmd.sat.date,i,namepixel), dpi=300, facecolor='w', edgecolor='w',
            orientation='landscape', format='pdf',
            transparent=False, bbox_inches='tight', pad_inches=0.1,
            metadata=None)
    plt.close()   
    
def PlotSsaMaskSnow(outfolder, osmd, i):
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    c1=ax1.imshow(osmd.model.ssa)
    plt.colorbar(c1)
    plt.title("ssa %s iter %s"%(osmd.sat.date,i))
    
    ax2 = fig.add_subplot(222)
    c2=ax2.imshow(osmd.model.a,vmin=0,vmax=5)#norm=LogNorm())
    plt.colorbar(c2)
    plt.title("a")
    
    ax3 = fig.add_subplot(223)
    c3=ax3.imshow(1./osmd.model.a*osmd.model.rmse,vmin=0.,vmax=0.55)
    # c3=ax3.imshow(err)#norm=LogNorm())
    plt.colorbar(c3)
    plt.title("1./a*rmse")
    
    ax4 = fig.add_subplot(224)
    c4=ax4.imshow(osmd.model.snowmask["isSnow"].values)
    # c4=ax4.imshow(err)#norm=LogNorm())
    plt.colorbar(c4)
    plt.title("mask: snow if ((1./a*rmse)<0.3) & (a<4)")
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    
    
    plt.show()    
    
    fig.savefig(outfolder+"%s%sMaskSnow.pdf"%(osmd.sat.date,i), dpi=300, facecolor='w', edgecolor='w',
        orientation='landscape', format='pdf',
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        metadata=None)
    plt.close()
    
def PlotHistGenerateMask(outfolder, osmd, i):
    fig = plt.figure()
    plt.hist((1./osmd.model.a*osmd.model.rmse).flatten(),100,alpha=0.5,histtype='bar',ec="black")
    
    plt.title("1./a*rmse %s iteration %s"%(osmd.sat.date,i))
    plt.show()
    fig.savefig(outfolder+"%s%sHistERR.pdf"%(osmd.sat.date,i), dpi=300, facecolor='w', edgecolor='w',
    orientation='landscape', format='pdf',
    transparent=False, bbox_inches='tight', pad_inches=0.1,
    metadata=None)
    plt.close()