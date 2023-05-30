import numpy as np
import os

from shapely.geometry import Point

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Polygon

from functions import get_swift_orbit
from swifttools.swift_too import Clock

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class BATSAA:
    def __init__(self, lat, long):
        self.lat = lat
        self.long = long
        self.points = np.load(os.path.join(BASE_DIR, 'static', 'np_files', 'saa_points.npy'))
        self.saapoly = Polygon(self.points)

    def insaa(self):
        # '''For a given time, are we inside the BAT SAA polygon'''
        # if not self.eph:
        #     self.eph = STKReadEph(latestephem(self.year,self.day))
        # index = self.eph.ephindex(utime)

        # self.long = self.eph.long[index]
        # self.lat = self.eph.lat[index]

        return self.saapoly.contains(Point(self.long, self.lat))


def calc_mcilwain_l(longitude, latitude):
    """

    Estimate the McIlwain L value given the latitude (-30, +30) and
    East Longitude.  This uses a cubic polynomial approximation to the full
    calculation and is similar to the approach used by the GBM FSW.

    Args:
        longitude (np.array): East longitude in degrees from 0 to 360
        latitude (np.array): Latitude in degrees from -180 to 180

    Returns:
        np.array: McIlwain L value

    """

    latitude = np.asarray([latitude])
    longitude = np.asarray([longitude])
    orig_shape = latitude.shape
    latitude = latitude.flatten()
    longitude = longitude.flatten()
    poly_coeffs = np.load(os.path.join(BASE_DIR, 'static', 'np_files', 'poly_coeffs.npy'))

    longitude[longitude < 0.0] += 360.0
    longitude[longitude == 360.0] = 0.0

    bad_idx = (latitude < -30.0) | (latitude > 30.0) | (longitude < 0.0) | (longitude >= 360.0)
    if np.sum(bad_idx) != 0:
        raise ValueError(f'Out of range coordinates for McIlwain L for {np.sum(bad_idx)} locations')

    idx = np.asarray((longitude / 10.0).astype(int))
    idx2 = np.asarray(idx + 1)
    idx2[idx2 >= 36] = 0
    idx2 = idx2.astype(int)

    longitude_left = 10.0 * idx
    f = (longitude - longitude_left) / 10.0  # interpolation weight, 0 to 1

    try:
        num_pts = len(latitude)
    except:
        num_pts = 1

    mc_l = np.zeros(num_pts)
    for i in range(num_pts):
        mc_l[i] = (1.0 - f[i]) * (
                poly_coeffs[idx[i], 0] + poly_coeffs[idx[i], 1] * latitude[i] +
                poly_coeffs[idx[i], 2] *
                latitude[i] ** 2 + poly_coeffs[idx[i], 3] * latitude[i] ** 3
            ) +\
                  f[i] * (
                          poly_coeffs[idx2[i], 0] + poly_coeffs[idx2[i], 1] *
                          latitude[i] + poly_coeffs[idx2[i], 2] *
                          latitude[i] ** 2 + poly_coeffs[idx2[i], 3] *
                          latitude[i] ** 3
                  )
    mc_l = mc_l.reshape(orig_shape)
    return np.squeeze(mc_l)


def mcilwain_map(lon_range, lat_range, ax, saa_mask=None, alpha=0.5, **kwargs):
    """

    Plot the McIlwain L heatmap on a Basemap

    Parameters:
    -----------
    lat_range: (float, float)
        The latitude range
    lon_range: (float, float)
        The longitude range
    ax: matplotlib.axes
        The plot axes references
    saa_mask: polygon, optional
        Masks out area of heatmap for SAA if given. None by default.
    alpha: float, optional
        The alpha opacity of the heatmap
    kwargs: optional
        Other plotting keywords

    Returns:
    -----------
    image: QuadContourSet
        The heatmap plot object

    """

    # Create an array on the earth
    lon_array = np.linspace(*lon_range, 720)
    lat_array = np.linspace(*lat_range, 108)
    LON, LAT = np.meshgrid(lon_array, lat_array)

    # Convert to Mercator
    mrc_ranges = ccrs.Mercator().transform_points(ccrs.PlateCarree(), LON, LAT)
    mLON = mrc_ranges[:, :, 0]
    mLAT = mrc_ranges[:, :, 1]
    # Calculate McIlwain L over the grid
    mcl = calc_mcilwain_l(LON, LAT)

    if saa_mask:
        saa_path = saa_mask.get_path()
        mask = saa_path.contains_points(
            np.array((mLON.ravel(), mLAT.ravel())).T)
        mcl[mask.reshape(mcl.shape)] = 0.0

    # do the plot
    levels = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
    image = ax.contourf(mLON, mLAT, mcl, levels=levels, alpha=alpha, **kwargs)

    return image


def earth_line(lat, lon):
    """Plot a line on the Earth (e.g. orbit)

    Parameters:
    -----------
    lat: np.array
        Array of latitudes
    lon: np.array
        Array of longitudes
    ind: int
        Index of lat and lon closest to the location of the spacecraft at trigger time

    Returns:
    -----------
    points: list
        List of numpy arrays that are the (x, y) coordinates of each segment in Mercator projection
    """

    lat = np.array(lat)
    lon = np.array(lon)

    lat[(lat > 180)] -= 360
    lon[(lon > 180)] -= 360

    path = np.vstack((lon, lat))
    isplit = np.nonzero(np.abs(np.diff(path[0])) > 5.0)[0]
    segments = np.split(path, isplit + 1, axis=1)

    return segments


def make_earthplot(trigtime, trigid, orbitdat, prompt=True, outdir=''):
    """Create and save plot of Swift's trajectory over earth

    Parameters:
    -----------
    trigtime: datetime.datetime
        Time of trigger
    trigid: str
        ID of the trigger for title
    times, sclats, sclons: lists of floats
        Lists grabbed from sao.fits file to chart spacecraft's trajectory
    prompt: bool
        Represents whether above lists are pulled from quicklook or archived TLE data
    outdir: str (optional)
        Directory to save image in

    Returns:
    -----------
    filename: str
        Name of earthplot file created in working directory
    """

    mrc = ccrs.Mercator()

    # Define extent of plot in longitude and latitude
    lon_range = np.array([-180.0, 180.0])
    lat_range = np.array([-30.0, 30.0])

    # Transform to Mercator
    mrc_range = mrc.transform_points(ccrs.PlateCarree(), lon_range, lat_range)

    fig, ax = plt.subplots(figsize=(20, 10), dpi=100, subplot_kw={'projection': mrc})
    ax.set_xlim(mrc_range[0, 0], mrc_range[1, 0])
    ax.set_ylim(mrc_range[0, 1], mrc_range[1, 1])

    ax.coastlines()

    # Add parallels and meridians
    ax.gridlines(draw_labels=True, xlocs=np.arange(-180., 181., 30.), ylocs=np.arange(-90., 91., 30.),
                 linewidth=1, linestyle='dotted', color='black')

    # Grab position of spacecraft at time of trigger
    spacecraft_time = Clock(utctime=trigtime).met

    if prompt:
        triglon, triglat = orbitdat.SAT_LON[1000], orbitdat.SAT_LAT[1000]
        index = 1000
    else:
        # Get position vectors within 1 second of target time
        index = np.where((orbitdat.TIME_ADJ < spacecraft_time+1) & (orbitdat.TIME_ADJ > spacecraft_time-1))
        points = orbitdat[index]

        triglon, triglat = points.SAT_LON[0], points.SAT_LAT[0]
        if triglon > 180:
            triglon = -(360 - triglon)
        if triglat > 180:
            triglat = -(360 - triglat)

    mrc_triglon, mrc_triglat = mrc.transform_point(triglon, triglat, ccrs.PlateCarree())

    im = plt.imread(os.path.join(BASE_DIR, 'static', 'saticon.png'))
    ab = AnnotationBbox(OffsetImage(im, zoom=1), (mrc_triglon, mrc_triglat), frameon=False)
    ax.add_artist(ab)

    # Plots line of trajectory of spacecraft around trigger time

    time_range = np.where((orbitdat.TIME_ADJ < spacecraft_time + 1000) & (orbitdat.TIME_ADJ > spacecraft_time - 500))
    trajectory = orbitdat[time_range]

    for segment in earth_line(trajectory.SAT_LAT.tolist(), trajectory.SAT_LON.tolist()):
        ax.scatter(segment[0, :], segment[1, :], transform=ccrs.PlateCarree(), s=1, c='purple', alpha=.5)

    # SAA polygon
    lons, lats = list(zip(*BATSAA(triglon, triglat).points))
    xy = mrc.transform_points(ccrs.PlateCarree(), np.array(lons), np.array(lats))[:, 0:2]
    saa_poly = Polygon(xy, facecolor=to_rgba('darkred', .4), edgecolor='darkred', linewidth=1.5, zorder=2)
    ax.add_patch(saa_poly)

    # McIlwain L
    artist = mcilwain_map(lon_range, lat_range, ax, saa_poly, alpha=0.5)
    cb = plt.colorbar(artist, label='McIlwain L', ax=ax, shrink=0.6, pad=0.05, orientation='horizontal')
    cb.draw_all()

    if prompt:
        title = f'Earthplot from TLE (provisional)'
        ax.set_title(title)
    else:
        title = f'Earthplot (final)'
        ax.set_title(title)

    filename = os.path.join(outdir, f'{trigid}_EARTHPLOT.png')
    fig.savefig(filename,bbox_inches='tight')

    plt.close(fig)

    return filename


def SwiftEarthPlot(trigtime, trigid, prompt=True, outdir='', orbitdat=None):
    if orbitdat is None:
        orbitdat = get_swift_orbit(trigtime)
    filename = make_earthplot(trigtime, trigid, orbitdat, prompt=prompt, outdir=outdir)

    return filename
