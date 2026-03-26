import numpy as np
import xarray as xr
import plotly.graph_objects as go
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature
import cartopy.crs as ccrs
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

# Create the contour plot using Plotly's scatter with contour-like appearance
# This approach uses scattered points with interpolation
fig = go.Figure()

# Method 1: Use Plotly's built-in polar coordinates
# Convert to polar coordinates (r = 90-lat, theta = lon)
r_vals = 90 - lats
theta_vals = lons

# Create a heatmap in polar coordinates
fig.add_trace(go.Scatterpolar(
    r=r_vals.flatten(),
    theta=theta_vals.flatten(),
    mode='markers',
    marker=dict(
        size=1.5,
        color=x_cyclic.flatten(),
        colorscale='viridis',
        cmin=-100,
        cmax=0,
        showscale=True,
        colorbar=dict(title='SWCF')
    ),
    showlegend=False
))

# Alternative method using regular contour plot with proper coordinate handling
# Comment out the above and uncomment below to try this approach:

# # Project to stereographic coordinates properly
# proj = ccrs.NorthPolarStereo()
# pc = ccrs.PlateCarree()
# 
# # Transform coordinates using cartopy (more accurate)
# x_proj = []
# y_proj = []
# z_proj = []
# 
# for i in range(lats.shape[0]):
#     for j in range(lats.shape[1]):
#         # Transform each point
#         transformed = proj.transform_point(lons[i,j], lats[i,j], pc)
#         x_proj.append(transformed[0])
#         y_proj.append(transformed[1])
#         z_proj.append(x_cyclic[i,j])
# 
# # Create contour plot with scattered data
# fig.add_trace(go.Scatter(
#     x=x_proj,
#     y=y_proj,
#     mode='markers',
#     marker=dict(
#         size=2,
#         color=z_proj,
#         colorscale='viridis',
#         cmin=-100,
#         cmax=0,
#         showscale=True,
#         colorbar=dict(title='SWCF')
#     ),
#     showlegend=False
# ))

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

# Extract coastline coordinates and convert to polar
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
            
            # Convert to polar coordinates
            coast_r = [90 - lat for lat in coast_lats_nh]
            coast_theta = coast_lons_nh
            
            fig.add_trace(go.Scatterpolar(
                r=coast_r,
                theta=coast_theta,
                mode='lines',
                line=dict(color='black', width=0.5),
                showlegend=False
            ))

# Set up polar layout
fig.update_layout(
    title="SWCF - North Polar Stereographic",
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 90],
            tickvals=[0, 30, 60, 90],
            ticktext=['90°N', '60°N', '30°N', '0°N']
        ),
        angularaxis=dict(
            direction='clockwise',
            period=360,
            tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
            ticktext=['0°', '45°E', '90°E', '135°E', '180°', '135°W', '90°W', '45°W']
        )
    ),
    width=800,
    height=800
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