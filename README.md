# REDRESS
Radiative transfEr in ruggeD teRrain for rEmote Sensing of Snow

**Abstract:**

The monitoring of snow-covered surfaces on Earth is largely facilitated by the wealth of satellite data available, with increasing spatial resolution and temporal coverage over the last years. Yet to date, retrievals of snow physical properties still remain complicated in mountainous areas, owing to the complex interactions of solar radiation with terrain features such as multiple scattering between slopes, exacerbated over bright surfaces. Existing physically-based models of solar radiation across rough scenes are either too complex and resource-demanding for the implementation of systematic satellite image processing, not designed for highly reflective surfaces such as snow, or tied to a specific satellite sensor. This study proposes a new formulation, combining a forward model of solar radiation over rugged terrain with dedicated snow optics into a flexible multi-sensor tool that bridges a gap in the optical remote sensing of snow-covered surfaces in mountainous regions. The model presented here allows to perform rapid calculations over large snow-covered areas. Good results are obtained even for extreme cases, such as steep shadowed slopes or on the contrary, strongly illuminated sun-facing slopes. Simulations of Sentinel-3 OLCI scenes performed over a mountainous region in the French Alps allow to reduce the bias by up to a factor 6 in the visible wavelengths compared to methods that account for slope inclination only. Furthermore, the study underlines the contribution of the individual fluxes to the total top-of-atmosphere radiance, highlighting the importance of reflected radiation from surrounding slopes which, in mid-winter after a recent snowfall  (13 February 2018), account on average for 7\% of the signal at 400 nm and 16\% at 1020 nm (on 13 February 2018), as well as coupled diffuse radiation scattered by the neighbourhood, that contributes to 18\% at 400 nm and 4\% at 1020 nm. Given the importance of these contributions, accounting for slopes and reflected radiation between terrain features is a requirement for improving the accuracy of satellite retrievals of snow properties over snow-covered rugged terrain. The forward formulation presented here is the first step toward this goal, paving the way for future retrievals.

---
**Install :**

install conda then :

conda config --set auto_activate_base false

conda config --add channels conda-forge

conda config --set channel_priority strict

conda env create -f package_list.yml

conda activate Redress


**Run**

All informations in "script_lancement.py"




Code coming soon.
