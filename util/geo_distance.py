
import numpy as np
#
# Geographic Centroid
#
def get_centroid(in_lat, in_lon):
    """Given collection of latitudes and longitudes,
       return the location (lon, lat) of the geographic centroid.
       
       in_lat :: list or array of latitudes in DEGREES.
       in_lon :: list or array of longitudes in DEGREES.
       
       return :: tuple (lat, lon) of the centroid
    """
    assert len(in_lat) == len(in_lon)
    if isinstance(in_lat, list):
        in_lat = np.array(in_lat)
    if isinstance(in_lon, list):
        in_lon = np.array(in_lon)
    # convert to spherical:
    xpts = np.cos(np.radians(in_lat)) * np.cos(np.radians(in_lon))
    ypts = np.cos(np.radians(in_lat)) * np.sin(np.radians(in_lon))
    zpts = np.sin(np.radians(in_lat))

    # average (not weighted):
    xctr = xpts.mean()
    yctr = ypts.mean()
    zctr = zpts.mean()

    # convert to lat lon
    ctr_lon = np.arctan2(yctr, xctr)
    hyp = np.sqrt(xctr**2 + yctr**2)
    ctr_lat = np.arctan2(zctr, hyp)
    return ctr_lat, ctr_lon


#
# Haversine Distance (note there is a package called haversine)
#
def distance(lat1, lon1, lat2, lon2, r=None):
    """
    Using Haversine function, get distance between 
    (lat1, lon1) and (lat2, lon2) in meters.
    
    lon1, lat1, lon2, lat2 :: coordinates in DEGREES
    r :: is the radius of the sphere; if None use Earth radius
    return :: distance in meters
    """
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    if r is None:
        r = 6378.1 * 1000. # km * (1000m/km) = meters
    term1 = (np.sin((lat2 - lat1)/2))**2
    term2 = np.cos(lat1)*np.cos(lat2)*(np.sin((lon2 - lon1) / 2))**2
    return 2 * r * np.arcsin(np.sqrt(term1 + term2))
    

