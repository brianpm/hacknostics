import numpy as np
import xarray as xr
import plotly.graph_objects as go
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from scipy.interpolate import griddata
import time

# Load your data (same as before)
fil = "/Users/brianpm/Documents/cesm2_lens_data/SWCF/b.e21.BHISTsmbb.f09_g17.LE2-1151.008.cam.h0.SWCF.188001-188912.nc"
ds = xr.open_dataset(fil).isel(time=100)
x = ds['SWCF']
x_nh = x.sel(lat=slice(0,90))

# Add cyclic point
x_cyclic, lon_cyclic = add_cyclic_point(x_nh, coord=x.lon)

# Create meshgrid
lons, lats = np.meshgrid(lon_cyclic, x_nh.lat)

# Define contour levels
levels = np.arange(-100, 0, 10)

# Start timing the plotting portion
start_time = time.time()

# Convert to stereographic projection coordinates using cartopy
proj = ccrs.NorthPolarStereo()
pc = ccrs.PlateCarree()

def latlon_to_stereo_cartopy(lat, lon):
    """Convert lat/lon to stereographic projection using cartopy"""
    # Handle arrays
    if np.isscalar(lat):
        transformed = proj.transform_point(lon, lat, pc)
        return transformed[0], transformed[1]
    else:
        # For arrays, transform each point
        x_coords = []
        y_coords = []
        lat_flat = lat.flatten()
        lon_flat = lon.flatten()
        
        for i in range(len(lat_flat)):
            transformed = proj.transform_point(lon_flat[i], lat_flat[i], pc)
            x_coords.append(transformed[0])
            y_coords.append(transformed[1])
        
        x_array = np.array(x_coords).reshape(lat.shape)
        y_array = np.array(y_coords).reshape(lat.shape)
        return x_array, y_array

# Project all coordinates
x_proj, y_proj = latlon_to_stereo_cartopy(lats, lons)

# Create regular grid in stereographic coordinates for interpolation
# Determine grid bounds
x_min, x_max = x_proj.min(), x_proj.max()
y_min, y_max = y_proj.min(), y_proj.max()

# Create regular grid
grid_resolution = 200  # Adjust for quality vs speed
xi = np.linspace(x_min, x_max, grid_resolution)
yi = np.linspace(y_min, y_max, grid_resolution)
xi_grid, yi_grid = np.meshgrid(xi, yi)

# Flatten arrays for interpolation
points = np.column_stack((x_proj.flatten(), y_proj.flatten()))
values = x_cyclic.flatten()

# Remove NaN values
valid_mask = ~np.isnan(values)
points_clean = points[valid_mask]
values_clean = values[valid_mask]

# Interpolate to regular grid
zi = griddata(points_clean, values_clean, (xi_grid, yi_grid), method='linear')

# Create the filled contour plot
fig = go.Figure()

# Add filled contours
fig.add_trace(go.Contour(
    x=xi,
    y=yi,
    z=zi,
    contours=dict(
        start=-100,
        end=0,
        size=10,
        coloring='fill'
    ),
    colorscale='viridis',
    showscale=True,
    colorbar=dict(title="SWCF"),
    line=dict(width=0)  # Remove contour lines, just filled
))

# Add coastlines using cartopy data
def extract_coords_from_geom(geom):
    """Extract coordinates from various geometry types"""
    coords_list = []
    
    if hasattr(geom, 'geoms'):  # MultiLineString, MultiPolygon, etc.
        for sub_geom in geom.geoms:
            coords_list.extend(extract_coords_from_geom(sub_geom))
    elif hasattr(geom, 'exterior'):  # Polygon
        coords_list.append(list(geom.exterior.coords))
    elif hasattr(geom, 'coords'):  # LineString, Point
        coords_list.append(list(geom.coords))
    
    return coords_list

coastlines = cfeature.COASTLINE.with_scale('50m')
coast_geoms = coastlines.geometries()

# Extract coastline coordinates and convert to stereographic
for geom in coast_geoms:
    coords_list = extract_coords_from_geom(geom)
    
    for coords in coords_list:
        if len(coords) < 2:
            continue
            
        coast_lons = [coord[0] for coord in coords]
        coast_lats = [coord[1] for coord in coords]
        
        # Only plot coastlines in Northern Hemisphere
        nh_indices = [i for i, lat in enumerate(coast_lats) if lat >= 0]
        
        if len(nh_indices) > 1:
            coast_lats_nh = [coast_lats[i] for i in nh_indices]
            coast_lons_nh = [coast_lons[i] for i in nh_indices]
            
            # Convert coastlines to stereographic coordinates
            coast_x, coast_y = latlon_to_stereo_cartopy(np.array(coast_lats_nh), np.array(coast_lons_nh))
            
            fig.add_trace(go.Scatter(
                x=coast_x,
                y=coast_y,
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False
            ))

# Add a circular boundary (approximate Arctic Circle or desired boundary)
theta_boundary = np.linspace(0, 2*np.pi, 100)
boundary_lat = np.full_like(theta_boundary, 20)  # 20N latitude boundary
boundary_lon = np.degrees(theta_boundary)
x_boundary, y_boundary = latlon_to_stereo_cartopy(boundary_lat, boundary_lon)

fig.add_trace(go.Scatter(
    x=x_boundary,
    y=y_boundary,
    mode='lines',
    line=dict(color='black', width=2),
    showlegend=False
))

# Set equal aspect ratio and clean layout
fig.update_layout(
    title="SWCF - North Polar Stereographic",
    xaxis=dict(
        scaleanchor="y",
        scaleratio=1,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[x_min, x_max]
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[y_min, y_max]
    ),
    width=800,
    height=800,
    plot_bgcolor='white'
)

# End timing
end_time = time.time()
print(f"Plotting time: {end_time - start_time:.3f} seconds")

# Save as PNG
print("Saving PNG...")
save_start = time.time()
fig.write_image("plotly_polar_contour.png", width=800, height=800, scale=2)
save_end = time.time()
print(f"PNG saved as 'plotly_polar_contour.png' in {save_end - save_start:.3f} seconds")