import os
import numpy as np

import ephem
from astropy import wcs
from astropy.io import fits
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from .functions import get_swift_orbit
from swifttools.swift_too import ObsQuery, Clock, PlanQuery

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


import ligo.skymap.plot
import ligo.skymap.plot.poly
import ligo.skymap.io.fits
import ligo.skymap.postprocess.util
import healpy as hp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Convert right ascension and declination (in degrees) to 3D unit vectors (returns x, y, z)
def ra_dec_to_uvec(ra, dec):
    phi = np.deg2rad(90 - dec)
    theta = np.deg2rad(ra)

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)

    return x, y, z


# Convert 3D Unit vectors to right ascension and declination (deg)
def uvec_to_ra_dec(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    x = x / r
    y = y / r
    z = z / r

    theta = np.arctan2(y, x)
    phi = np.arccos(z)
    dec = 90 - np.rad2deg(phi)
    if theta < 0:
        ra = 360 + np.rad2deg(theta)
    else:
        ra = np.rad2deg(theta)

    return ra, dec


# This class was involved in preprocessing and is not called during regular usage
class SwiftBatPartialCoding:
    # Note: this class is copied with limited adjustments from BAT-tools/swift_poshist
    # https://github.com/Tohuvavohu/BAT-tools/blob/9f506c5382272f73546e2ea29147df76deae5f5e/swift_poshist/swift_poshist.py
    _file = os.path.join(BASE_DIR, 'static', 'pcode_default.img')
    nside = 128

    def __init__(self):
        hdulist = fits.open(self._file, memmap=False)
        w = wcs.WCS(hdulist[0].header)
        data = hdulist[0].data

        num_y, num_x = w.array_shape
        x = np.arange(num_x)
        y = np.arange(num_y)
        x, y = np.meshgrid(x, y)
        ra, dec = w.wcs_pix2world(x, y, 1)
        ra += 360.0

        npix = hp.nside2npix(self.nside)
        pix = hp.ang2pix(self.nside, ra, dec, lonlat=True)
        self._hpx = np.zeros(npix)
        self._hpx[pix] = data.reshape(pix.shape)

    def partial_coding_path(self, frac, numpts_ra=360, numpts_dec=180):
        """Return the bounding path for a given partial coding fraction. This is the method that created
        the bat_foot.npy file, but this was preprocessing and the method is not called during regular
        operation.

        Args:
            frac (float): The partial coding fraction (valid range 0-1)
            numpts_ra (int, optional): The number of grid points along the RA
                                       axis. Default is 360.
            numpts_dec (int, optional): The number of grid points along the Dec
                                        axis. Default is 180.

        Returns:
            [(np.array, np.array), ...]: A list of RA, Dec points, where each \
                item in the list is a continuous closed path.
        """
        # create the grid and integrated probability array
        grid_pix, phi, theta = self._mesh_grid(numpts_ra, numpts_dec)
        frac_arr = self._hpx[grid_pix]
        ra = self._phi_to_ra(phi)
        dec = self._theta_to_dec(theta)

        # use matplotlib contour to produce a path object
        fracs = [frac] + [x for x in np.concatenate((np.arange(0, .9, .01),
                                                     np.array([.95, .99, .999]))) if x > frac]
        contour = plt.contour(ra, dec, frac_arr, fracs)

        polys_dict = {fracs[i]: [] for i, x in enumerate(contour.allsegs)}
        nrow = 0
        for i, x in enumerate(contour.allsegs):
            level_polys = []
            for poly in x:
                nrow += len(poly)
                level_polys.append(poly)
            polys_dict[fracs[i]] = np.concatenate(level_polys)

        polys = np.zeros((nrow, 3))
        first_row = 0
        for level, poly in polys_dict.items():
            polys[first_row:(first_row+len(poly))] = np.vstack(
                (poly[:, 0],
                 poly[:, 1],
                 np.repeat(level, len(poly)))
            ).T
            first_row += len(poly)

        # np.save('bat_foot.npy', polys)

        # unfortunately matplotlib will plot this, so we need to remove
        for c in contour.collections:
            c.remove()

        return polys

    def _mesh_grid(self, num_phi, num_theta):
        # create the mesh grid in phi and theta
        theta = np.linspace(np.pi, 0.0, num_theta)
        phi = np.linspace(0.0, 2 * np.pi, num_phi)
        phi_grid, theta_grid = np.meshgrid(phi, theta)
        grid_pix = hp.ang2pix(self.nside, theta_grid, phi_grid)
        return grid_pix, phi, theta

    @staticmethod
    def _phi_to_ra(phi):
        return np.rad2deg(phi)

    @staticmethod
    def _theta_to_dec(theta):
        return np.rad2deg(np.pi / 2.0 - theta)


def get_swift_pos(datetime, orbitdat=None):
    if orbitdat is None:
        orbitdat = get_swift_orbit(datetime)

    spacecraft_time = Clock(utctime=datetime).met
    # Get position vectors within 1 second of target time
    index = np.where((orbitdat.TIME_ADJ < spacecraft_time+1) & (orbitdat.TIME_ADJ > spacecraft_time-1))
    points = orbitdat[index]

    # Take the first of each, since multiple can be within the time range
    lon, lat, alt = points.SAT_LON[0], points.SAT_LAT[0], points.SAT_ALT[0]
    if lon > 180:
        lon = -(360 - lon)
    if lat > 180:
        lat = -(360 - lat)

    return lon, lat, alt


def deg2dm(deg):
    sign = np.sign(deg)
    deg = np.abs(deg)
    d = np.floor(deg)
    m = (deg - d) * 60
    return int(sign * d), m


def get_geocenter(datetime, lon, lat):
    # Define the observer to be at the location of the spacecraft
    observer = ephem.Observer()

    # Convert the longitude to +E (-180 to 180)
    if lon > 180:
        lon = (lon % 180) - 180

    lon_deg, lon_min = deg2dm(lon)
    lat_deg, lat_min = deg2dm(lat)

    lon_string = '%s:%s' % (lon_deg, lon_min)
    lat_string = '%s:%s' % (lat_deg, lat_min)

    observer.lon = lon_string
    observer.lat = lat_string

    # Set the time of the observations
    observer.date = ephem.date(datetime)

    # Get the ra and dec (in radians) of the point in the sky at altitude = 90 (directly overhead)
    ra_zenith_radians, dec_zenith_radians = observer.radec_of('0', '90')

    # convert the ra and dec to degrees
    ra_zenith = np.degrees(ra_zenith_radians)
    dec_zenith = np.degrees(dec_zenith_radians)

    ra_geocenter = (ra_zenith + 180) % 360
    dec_geocenter = -1 * dec_zenith

    return ra_geocenter, dec_geocenter


def get_earth_sat_pos(datetime, orbitdat=None):
    lon, lat, altitude = get_swift_pos(datetime, orbitdat=orbitdat)

    # Get the geo center coordinates in ra and dec
    ra_geocenter, dec_geocenter = get_geocenter(datetime, lon, lat)

    EARTH_RADIUS = 6378.140
    occultation_radius = np.rad2deg(np.arcsin(EARTH_RADIUS / altitude))

    return ra_geocenter, dec_geocenter, occultation_radius


def make_earth_contour(radius):
    thetas = np.linspace(0, -2 * np.pi, 200)
    radii = [r for r in np.arange(0, 90, 5) if r < radius] + [radius]

    ras = np.concatenate([r * np.cos(thetas) for r in radii])
    decs = np.concatenate([r * np.sin(thetas) for r in radii])

    earth_poly = np.c_[ras, decs]

    return earth_poly


class Footprint:
    # Parent class for BatFootprint and EarthFootprint
    def __init__(self, poly):
        self.original_poly = poly
        self.projected_poly = None
        self.foot_pos = None
        self.split_polys = None
        self.final_polys = None

    @staticmethod
    def _split_polygon(poly):
        # Splits polygon into two if...
        poly = poly[np.argsort(poly[:, 0], axis=0)]
        jump_ind = np.argmax(poly[1:, 0] - poly[0:-1, 0])

        # ...the largest jump between two RA values is >50, or...
        if (poly[1:, 0] - poly[0:-1, 0])[jump_ind] > 50:
            hi = poly[jump_ind+1:]
            lo = poly[0:jump_ind+1]
            return [x for x in [hi, lo] if len(x) > 3]  # at least 3 points

        # ...the polygon spans nearly the entire map with no jumps, which indicates that it's on the edge.
        # In this case, we find a split by gradually filtering out dec values until there is such a jump in RA,
        # then splitting into 3 polys, one for each side of the jump and one for the center.
        elif np.max(poly[:, 0]) - np.min(poly[:, 0]) > 350:
            cutoff = 89
            filtered_poly = poly[abs(poly[:, 1]) < cutoff]
            jump_ind = np.argmax(filtered_poly[1:, 0] - filtered_poly[0:-1, 0])
            while (filtered_poly[1:, 0] - filtered_poly[0:-1, 0])[jump_ind] < 50:
                filtered_poly = poly[abs(poly[:, 1]) < cutoff]
                try:
                    jump_ind = np.argmax(filtered_poly[1:, 0] - filtered_poly[0:-1, 0])
                except ValueError:
                    # If we have an empty array, it's because the poly did not need to be split
                    return [poly]
                cutoff -= 5

            # Get the index of the same points in the original polygon
            # analogous to jump_ind
            start = np.argwhere(
                (poly[:, 0] == filtered_poly[jump_ind, 0]) & (poly[:, 1] == filtered_poly[jump_ind, 1])
            )[0][0]
            # analogous to jump_ind + 1
            stop = np.argwhere(
                (poly[:, 0] == filtered_poly[jump_ind + 1, 0]) & (poly[:, 1] == filtered_poly[jump_ind + 1, 1])
            )[0][0]

            hi = poly[stop:]
            lo = poly[0:start+1]
            mid = poly[start+1:stop]

            return [x for x in [hi, lo, mid] if len(x) > 3]  # at least 3 points
        else:
            return [poly]

    @staticmethod
    def _grab_hull(poly):
        # Gets vertices of convex hull of polygon, then interpolates points along each edge
        hull = ConvexHull(poly)
        points = poly[hull.vertices]

        interp_points = []
        for i in range(len(points)):
            xs = np.linspace(points[i, 0], points[(i+1) % len(points), 0], 50)
            ys = np.linspace(points[i, 1], points[(i+1) % len(points), 1], 50)
            interp_points.append(np.c_[xs, ys])

        return np.concatenate(interp_points)

    def _project_footprint(self, ra, dec, pos_angle):
        zero_center_ra = self.original_poly[:, 0]
        zero_center_dec = self.original_poly[:, 1]
        zero_center_uvec = [ra_dec_to_uvec(zero_center_ra[i], zero_center_dec[i])
                            for i in range(len(self.original_poly))]

        # Create rotation
        rotation = R.from_euler('XYZ', [-pos_angle, dec, -ra], degrees=True)

        proj_footprint = np.empty((len(zero_center_ra), 2))
        for i in range(len(zero_center_uvec)):
            x, y, z = zero_center_uvec[i]
            new_x, new_y, new_z = rotation.apply([x, y, z], inverse=True)
            pt_ra, pt_dec = uvec_to_ra_dec(new_x, new_y, new_z)
            proj_footprint[i, :] = [pt_ra, pt_dec]

        return proj_footprint

    def make_projection(self, ra, dec, roll):
        self.foot_pos = (ra, dec, roll)
        self.projected_poly = self._project_footprint(ra, dec, 360 - roll)

        self.split_polys = self._split_polygon(self.projected_poly)
        self.final_polys = [self._grab_hull(poly) for poly in self.split_polys]

        return self.final_polys


def make_skyplot(trigid, trigtime, fracs=(0.01, 0.1, 0.5), external_trig_data=None, outdir='', orbitdat=None,prompt=False):
    """Create and save plot of sky with the BAT footprint, earth occultation, and optional external data

    Parameters:
    -----------
    trigid: str
        ID of the trigger for title
    trigtime: datetime
        Time of the trigger
    fracs: tuple(float), optional
        The partial coverage fractions you want plotted
    external_trig_data: (np.ndarray, dict), optional
        External trigger data and metadata, e.g. from Fermi
    outdir: str, optional
        Directory to save image in

    Returns:
    -----------
    filename: str
        Name of skyplot file created in working directory
    """

    # Create figure
    fig = plt.figure(figsize=(15, 15), dpi=100)

    ax = plt.axes(projection='astro degrees mollweide')
    ax.grid()

    if external_trig_data is not None:
        skymap, metadata = external_trig_data
        # Calculate probability per square degree
        nside = hp.get_nside(skymap)
        pixel_area_per_square_deg = hp.nside2pixarea(nside, degrees=True)
        prob_per_square_deg = skymap * pixel_area_per_square_deg

        cls = 100 * ligo.skymap.postprocess.util.find_greedy_credible_levels(skymap)
        vmax = prob_per_square_deg.max()

        # Mask to get rid of yucky brown void of low probability
        cutoff = np.quantile(prob_per_square_deg, .99)
        prob_per_square_deg[prob_per_square_deg < cutoff] = np.nan

        # Plot probability contours
        ax.imshow_hpx((prob_per_square_deg, 'ICRS'), nested=metadata['nest'], vmin=cutoff, vmax=vmax, cmap='cylon',
                      zorder=9, alpha=.6)
        ax.contour_hpx((cls, 'ICRS'), nested=metadata['nest'], colors='black', linewidths=2, 
                       levels=(50, 90,99), zorder=10)
        line1 = Line2D([0], [0], label='50,90,99% contour', color='black')

    geo_ra, geo_dec, geo_radius = get_earth_sat_pos(trigtime, orbitdat=orbitdat)
    geo = Footprint(make_earth_contour(geo_radius))
    earth_polys = geo.make_projection(geo_ra, geo_dec, 0)

    for poly in earth_polys:
        ax.fill(poly[:, 0], poly[:, 1], facecolor='blue', edgecolor='blue', linewidth=2, closed=True,
                transform=ax.get_transform('fk5'), zorder=2, alpha=.4, label='Earth')
    
    if prompt:
        oq = PlanQuery(begin=trigtime)[0]
        swift_ra, swift_dec, swift_roll = oq.ra, oq.dec, oq.roll
    else:
        oq = ObsQuery(begin=trigtime)[0]
        swift_ra, swift_dec, swift_roll = oq.ra, oq.dec, oq.roll

    bat = np.load(os.path.join(BASE_DIR, 'static', 'np_files', 'bat_foot.npy'))

    for frac in sorted(fracs, reverse=True):
        # Check if the frac given is a level in the pre-made file and note that rounding is occurring if not
        if not any(bat[:, 2] == frac):
            print(Warning(f"Requested coverage proportion {frac} not available; rounding to {np.min(bat[bat[:, 2] >= frac, 2])}."))

        sc = bat[bat[:, 2] >= frac]

        foot = Footprint(sc)
        swift_polys = foot.make_projection(swift_ra, swift_dec, 360 - swift_roll)

        for poly in swift_polys:
            ax.fill(poly[:, 0], poly[:, 1], facecolor='grey', edgecolor='black', closed=True,
                    transform=ax.get_transform('fk5'), zorder=3, alpha=.4, label='BAT FoV (50,10,1% coded)')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend1=plt.legend(by_label.values(), by_label.keys(),loc=1)
    if external_trig_data is not None:
        newhandles=[]
        newhandles.extend([line1])
        plt.legend(handles=newhandles,loc=2)
    plt.gca().add_artist(legend1)

    filename = os.path.join(outdir, f'{trigid}_SKYPLOT.png')
    fig.savefig(filename,bbox_inches='tight')

    plt.close(fig)

    return filename
