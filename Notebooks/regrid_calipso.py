import numpy as np
import xarray as xr
import xesmf as xe  # <-- xesmf is a nice wrapper for regridding; requires ESMpy
from pathlib import Path
import logging
logging.basicConfig(level=logging.DEBUG)



def load_var(v, d):
    fils = d[v]
    if len(fils) == 1:
        return xr.open_dataset(fils[0], decode_times=False)
    else:
        return xr.open_dataset(fils, decode_times=False, combine='by_coords')

    
logging.debug("Starting.")

# all the variables that come out of COSP (monthly),
# CAM NAME, CMIP NAME
cosp_master_list = [
    'CFAD_DBZE94_CS', #  , cfadDbze94
    'CFAD_SR532_CAL', #  , cfadLidarsr532
    'CLDHGH_CAL',     #  , clhcalipso
    'CLDHGH_CAL_ICE',
    'CLDHGH_CAL_LIQ',
    'CLDHGH_CAL_UN',
    'CLDLOW_CAL',     #  , cllcalipso
    'CLDLOW_CAL_ICE',
    'CLDLOW_CAL_LIQ',
    'CLDLOW_CAL_UN',
    'CLDMED_CAL',     #  , clmcalipso
    'CLDMED_CAL_ICE',
    'CLDMED_CAL_LIQ',
    'CLDMED_CAL_UN',
    'CLDTOT_CAL',     #  , cltcalipso
    'CLDTOT_CALCS',
    'CLDTOT_CAL_ICE',
    'CLDTOT_CAL_LIQ',
    'CLDTOT_CAL_UN',
    'CLDTOT_CS',
    'CLDTOT_CS2',
    'CLDTOT_ISCCP',   #  , cltisccp
    'CLD_CAL',        #  , clcalips
    'CLD_CAL_ICE',    #  , clcalipsoice
    'CLD_CAL_LIQ',    #  , clcalipsoliq
    'CLD_CAL_NOTCS',  #  , clcalipso2
    'CLD_CAL_UN',
    'CLD_MISR',       #  , clmisr
    'CLHMODIS',
    'CLIMODIS',       #  , climodis
    'CLLMODIS',
    'CLMMODIS',
    'CLMODIS',
    'CLRIMODIS',      #  , jpdftaureicemodis
    'CLRLMODIS',      #  , jpdftaureliqmodis
    'CLTMODIS',       #  , cltmodis
    'CLWMODIS',       #  , clwmodis
    'FISCCP1_COSP',   #  , clisccp
    'IWPMODIS',
    'LWPMODIS',
    'MEANCLDALB_ISCCP', #  , albisccp
    'MEANPTOP_ISCCP',   #  , pctisccp
    'MEANTAU_ISCCP',
    'MEANTBCLR_ISCCP',
    'MEANTB_ISCCP',
    'PCTMODIS',
    'REFFCLIMODIS',
    'REFFCLWMODIS',
    'RFL_PARASOL',     #  , parasolRefl
    'TAUILOGMODIS',
    'TAUIMODIS',
    'TAUTLOGMODIS',
    'TAUTMODIS',
    'TAUWLOGMODIS',
    'TAUWMODIS']


# MODEL DATA

# start with a single case:
data_loc = Path('/glade/collections/cdg/timeseries-cmip6')
case_name = 'f.e21.F1850_BGC.f09_f09_mg17.CFMIP-piSST.001'
# '/glade/collections/cdg/timeseries-cmip6/f.e21.F1850_BGC.f09_f09_mg17.CFMIP-piSST.001/atm/proc/tseries/month_1'
case_fils = sorted(list((data_loc / case_name / 'atm/proc/tseries/month_1').glob('*.nc')))
logging.info(f"Found {len(case_fils)} files in this location.")

# build a dictionary:
# key: variable name
# value: list of time series files

cosp_dict = dict()
for v in cosp_master_list:
    cosp_var_fils = []
    for f in case_fils:
        fil_var = f.stem.split('.')[-2]
        if v == fil_var:
            cosp_var_fils.append(f)
        if len(cosp_var_fils) > 0:
            cosp_dict[v] = cosp_var_fils

    
ds_to_merge = [load_var(c, cosp_dict) for c in ['CLDTOT_CAL', 'CLTMODIS', 'CLDTOT_ISCCP']]
ds = xr.merge(ds_to_merge)

logging.info("Model data done.")

# now let's start doing something.
# Get the CALIPSO data and compare long-term means
ds_obs = xr.open_mfdataset("/glade/work/brianpm/observations/clcalipso/MapLowMidHigh330m_*.nc", combine='by_coords')
# could automate this by trimming: 
obs_avg = ds_obs['cllcalipso'].sel(time=slice('2007-01-01', '2017-12-31')).mean(dim='time')

logging.info("Obs avg done")
print(obs_avg)


# I CAN NOT GET THIS TO WORK ON CASPER
# Seems like something wrong with ESMF installation,
# but I couldn't figure it out.
# let's try 2d interpolation ... BUT NOTE THAT WE NEED TO SHIFT LONGIUTDES
nlon = len(obs_avg['longitude'])
new_lon = (obs_avg['longitude'] + 360) % 360
# change the coordinate values and roll the data to get 1-360:
obs_avg_roll = obs_avg.copy(deep=True)
obs_avg_roll = obs_avg_roll.assign_coords(longitude=new_lon)
obs_avg_roll = obs_avg_roll.roll(longitude=nlon//2, roll_coords=True)

# for sanity, let's rename the coordinates:
obs_avg_roll = obs_avg_roll.rename({"latitude":"lat", "longitude":"lon"})

obs_avg_roll = obs_avg_roll.compute()

grid_out = xr.Dataset({'lat': ds['lat'].compute(), 'lon': ds['lon'].compute()})
logging.debug("Make the regridder.")
# Regridder(ds_in, ds_out, method, periodic=False, filename=None, reuse_weights=False)
regridder = xe.Regridder(obs_avg_roll, grid_out, 'bilinear', periodic=True, filename="/glade/work/brianpm/observations/clcalipso/calipso_to_fv_map.nc", reuse_weights=True)
print(regridder)
logging.debug("Done.")
