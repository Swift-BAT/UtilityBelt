import datetime
from astropy.time import Time
import time
import os
from astropy.io import fits
import numpy as np
from swifttools.swift_too import Data, Clock, ObsQuery
from urllib.request import urlretrieve
from ligo.gracedb.rest import GraceDb


def mjdtounix(mjd):
    # Return a floating point UNIX timestamp for a given MJD
    return Time(mjd, format="mjd", scale="utc").unix


def tjdsodtounix(tjd, sod):
    # Convert TJD and SOD pair to unix timestamp
    return mjdtounix(tjd + 40000) + sod


def tjdsodtodt(tjd, sod):
    # Convert TJD and SOD pair to datetime
    return datetime.datetime.fromtimestamp(tjdsodtounix(tjd, sod))


def unixtoutcdt(utime):
    # Return a MySQL format DATETIME string (UTC) for a given unix timestamp
    year, month, day, hour, minute, second, null, null, null = time.gmtime(utime)
    return datetime.datetime(year, month, day, hour, minute, second)


def unixtime2sc(unixtime):
    # Convert Unix time to Spacecraft time
    return (
        int(unixtime)
        - time.mktime((2001, 1, 1, 0, 0, 0, 0, 0, 0))
        + (unixtime - int(unixtime))
    )


def sctime2unix(sctime):
    # Convert Spacecraft time to Unix time
    return (
        int(sctime)
        + time.mktime((2001, 1, 1, 0, 0, 0, 0, 0, 0))
        + (sctime - int(sctime))
    )


def clock_correction(utime):
    c = Clock(utime)
    return c.utcf


def unixtime2ccmet(utime):
    return Clock(utime).met


def isInt(i):
    return isinstance(i, int)


def isFloat(i):
    return isinstance(i, float)


"""
Functions for Grabbing TLE Data From SWIFT
"""


def query_swift(trigtime, filename_suffix, **kwargs):
    # Input time of observation and filename following obsid (e.g. 'pat.fits.gz'), returns name of downloaded file
    obsid = None

    if obsid is None:
        # Failing over to ObsQuery for obsid - Note, this can be wrong for GUANO dumps that worked
        afst = ObsQuery(begin=trigtime)
        if len(afst) == 1:
            obsid = afst[0].obsid
    if obsid is None:
        raise ConnectionError(
            "Failed to get observation id from Swift API. Are you sure the trigger time is correct?"
        )

    d = Data()
    d.obsid = obsid
    d.bat = True
    d.match = f"*{filename_suffix}"

    if d.submit():
        if len(d.entries) == 1:
            urlretrieve(d[0].url, d[0].filename)
            return d[0].filename
        elif len(d.entries) > 1:
            if len(d.entries) < 5:
                raise ValueError(
                    f"Ambiguous suffix matched {[f.filename for f in d.entries]}, download manually instead."
                )
            else:
                raise ValueError(
                    f"Ambiguous suffix matched {len(d.entries)} files, download manually instead."
                )
        else:
            raise ValueError(f"No files matching suffix '{filename_suffix}' found.")


def get_swift_orbit(trigtime, recursed=False):
    # Get ID of observation at trigger time in order to get associated SAO file
    filepath = query_swift(trigtime, "sao.fits.gz")

    # Extract necessary info
    with fits.open(filepath) as sao_file:
        orbitdat = sao_file[1].data

    # Clean up
    os.remove(filepath)

    # Check that orbitdat grabbed contains the trigtime, since on occasion a trigtime on the border between
    # two events can be mistakenly associated with the preceding event. Also makes sure that it doesn't recurse
    # more than once.
    if min(orbitdat.TIME_ADJ) > Clock(utctime=trigtime).met and not recursed:
        return get_swift_orbit(trigtime - datetime.timedelta(seconds=60), recursed=True)
    if max(orbitdat.TIME_ADJ) < Clock(utctime=trigtime).met and not recursed:
        return get_swift_orbit(trigtime + datetime.timedelta(seconds=60), recursed=True)

    return orbitdat


def get_swift_attitude(trigtime):
    # Downloads file corresponding to trigtime, returns name of downloaded file
    filepath = query_swift(trigtime, "sat.fits.gz")

    # Extract necessary info
    with fits.open(filepath) as sat_file:
        att_data = sat_file[1].data
        utcf = sat_file[1].header["utcfinit"]

    # Clean up
    os.remove(filepath)

    return att_data, utcf


def get_64ms_rates(trigtime):
    # Downloads file corresponding to trigtime, returns name of downloaded file
    filepath = query_swift(trigtime, "rtms.lc.gz")

    # Extract necessary info
    with fits.open(filepath) as sat_file:
        rate_data = sat_file[1].data
        utcf = sat_file[1].header["utcfinit"]

    # Clean up
    os.remove(filepath)

    return rate_data, utcf


def get_quad_rates(trigtime):
    # Downloads file corresponding to trigtime, returns name of downloaded file
    filepath = query_swift(trigtime, "qd.lc.gz")

    # Extract necessary info
    with fits.open(filepath) as sat_file:
        rate_data = sat_file[1].data
        utcf = sat_file[1].header["utcfinit"]

    # Clean up
    os.remove(filepath)

    return rate_data, utcf


def findthehpxmap(trigtime):
    midnight = trigtime.replace(hour=0, minute=0, second=0, microsecond=0)
    sod = (trigtime - midnight).seconds
    fod = np.round(sod / 86400, 3)
    fodstr = str(fod)[2:]

    bntrig = trigtime.strftime("%y%m%d") + fodstr
    url = (
        "https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/"
        + str(trigtime.year)
        + "/bn"
        + bntrig
        + "/quicklook/glg_healpix_all_bn"
        + bntrig
        + ".fit"
    )

    return url


def findtheigwnmap(trigname):
    if trigname.startswith("S") is False:
        print("Not a superevent.")
        return False
    client = GraceDb()
    try:
        response=client.superevent(trigname)
    except:
        print('No such superevent on public GraceDB')
        return False
    if 'SKYMAP_READY' in response.json()['labels']:
        response=client.files(trigname)
        if 'Bilby.fits.gz' in response.json():
            response=client.files(trigname,'Bilby.fits.gz')
            skymap=response.read()
            filename='Bilby.fits.gz'
            skymapfile=open(filename,'wb')
            skymapfile.write(skymap)
            return filename
        elif 'bayestar.fits.gz' in response.json():
            response=client.files(trigname,'bayestar.fits.gz')
            skymap=response.read()
            filename='bayestar.fits.gz'
            skymapfile=open(filename,'wb')
            skymapfile.write(skymap)
            return filename
        elif 'cwb.fits.gz' in response.json():
            response=client.files(trigname,'cwb.fits.gz')
            skymap=response.read()
            filename='cwb.fits.gz'
            skymapfile=open(filename,'wb')
            skymapfile.write(skymap)
            return filename
        else:
            print('SKYMAP_READY flag set but no Bilby or bayestar or cwb map found.')
            return False
    else:
        print('Skymap not ready.')
        return False
