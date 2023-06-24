from ligo.skymap.postprocess import crossmatch
from astropy.table import QTable
from ligo.gracedb.rest import GraceDb

def cont_lev(coord,skymap)):

    coords_list = coord[n][1:-1].split(',')
    c = [float(coord.strip()) for coord in coords_list]
    c = ["%s  %s" %(c[0],c[1])]
    c = SkyCoord(c, unit=(u.deg, u.deg))

    
    return crossmatch(skymap, c).searched_prob



def I_omega(c,skymap):
  
    skymap = QTable.read(skymap)
    
    max_level = 29
    max_nside = ah.level_to_nside(max_level)
    level, ipix = ah.uniq_to_level_ipix(skymap['UNIQ'])
    index = ipix * (2**(max_level - level))**2
    
    sorter = np.argsort(index)
    ra, dec = c

    match_ipix = ah.lonlat_to_healpix(ra, dec, max_nside, order='nested')

    i = sorter[np.searchsorted(index, match_ipix, side='right', sorter=sorter)
            
    return 4 * np.pi * skymap[i]['PROBDENSITY'].to_value(u.rad**-2)




def J_FAR (gw_far, grb_far, I_omega, graceid):

    g = GraceDb()

    record = gc.superevent(graceid).json()
    gw_far = record['far']
    
    type = record['preferred_event_data']['group']

    Z_max = 30 * 3e-3 * 2 / (24 * 3600)
    Z = 30 * gw_far * grb_far

    if type == 'CBC':
        trials = 20 
        far_alert_thresh = 1 / (30 * 86400)

    elif type == 'Burst':
        trials = 12
        far_alert_thresh = 1 / (365 * 86400)

    joint_far = trials * Z * (1+ np.log(Z_max/Z)) /  I_omega

    return joint_far , joint_far / far_alert_thresh < 1
