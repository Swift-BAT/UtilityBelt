import matplotlib.pyplot as plt
import pandas as pd
import logging
import numpy as np
from datetime import datetime, timedelta


from gbm.data import HealPix
from gbm.data import GbmHealPix

from matplotlib.colors import LogNorm
import ligo.skymap.plot
import ligo.skymap.io
from astropy.table import Table
import astropy_healpix as ah
import healpy as hp
from ligo.gracedb.rest import GraceDb
import sys
sys.path.append('/gpfs/group/jak51/default/UtilityBelt/UtilityBelt')
from skyplot import get_earth_sat_pos
import scipy.interpolate
from scipy.spatial import Delaunay
from gbm.finder import TriggerCatalog
import os

from datetime import datetime
from astropy.time import Time


def down(graceid,file_name,path):

    client = GraceDb()
    response = client.files(graceid, file_name)
    skymap = response.read()
    name = f'{file_name}'
    file_path = os.path.join(path, name)
    with open(file_path, 'wb') as binary_file:
        binary_file.write(skymap)



class LinearNDInterpolatorExt(object):

    def __init__(self, points,values):
        self.funcinterp=scipy.interpolate.LinearNDInterpolator(points,values)
        self.funcnearest=scipy.interpolate.NearestNDInterpolator(points,values)
    def __call__(self,*args):
        t=self.funcinterp(*args)
        if not np.isnan(t):
            return t.item(0)
        else:
            return self.funcnearest(*args)
        

def f_ul(ra,dec,ras,decs,z_int):
    x = ras  
    y = decs 

    tri = Delaunay(np.column_stack((x, y)))        

    interp = LinearNDInterpolatorExt(tri, z_int)

    return interp(ra,dec)
 


def ang_sep(ra0, dec0, ra1, dec1):
    dcos = np.cos(np.radians(np.abs(ra0 - ra1)))
    angsep = np.arccos(
        np.cos(np.radians(90 - dec0)) * np.cos(np.radians(90 - dec1))
        + np.sin(np.radians(90 - dec0)) * np.sin(np.radians(90 - dec1)) * dcos
    )
    return np.rad2deg(angsep)
        

def extract_final_part(input_string):

    last_slash_index = input_string.rfind('/')

    final_part = input_string[last_slash_index + 1:]

    return final_part


def plot_ul_map(ras,decs,z,skymap_file, ra_ea, red_ea, radius_ear,name, path_results):

    #logging.info('check plot')

    def f_ul_int(x,y):
        return f_ul(x,y,ras,decs,z)


    #---- save the map as image


    nside = 32  
    npix = hp.nside2npix(nside)  # Total number of pixels in the map

    map_values = np.zeros(npix)  # Initialize the array with zeros  
    minval=np.inf
    for n in range(npix):

        theta, phi = hp.pix2ang(nside, n)  # Retrieve theta and phi angles
        dec = np.degrees(np.pi / 2 - theta)  # Convert theta to Dec
        ra = np.degrees(phi)  # Convert phi to RA

        
        if ang_sep(ra_ea, red_ea, ra, dec)> radius_ear:

            map_values[n]=f_ul_int(ra,dec)
            minval=min(map_values[n],minval)
        
        else:
            map_values[n]='nan'


    hp.write_map(os.path.join(path_results,'ul.fits'), map_values, coord='C', overwrite=True)


    hpx, header = hp.read_map(os.path.join(path_results,'ul.fits'), h=True)

    fig = plt.figure(figsize=(15, 11), dpi=100)

    ax = plt.axes(projection='astro degrees mollweide')
    ax.grid()

    if skymap_file!='None':

        if skymap_file[0]=='point':

            ax.plot(skymap_file[1], skymap_file[2], transform=ax.get_transform('world'), marker='x' , c='red', zorder=1, ms='20', markeredgewidth=3)

        else:
            # Plot probability contours
            skymap, metadata = ligo.skymap.io.fits.read_sky_map(skymap_file, nest=True, distances=False)

            cls = 100 * ligo.skymap.postprocess.util.find_greedy_credible_levels(skymap)

            ax.contour_hpx((cls, 'ICRS'), nested=metadata['nest'], colors='black', linewidths=1.5, levels=(50, 90), zorder=10, linestyles=['dashed', 'solid'])
    

    # Create a scalar mappable for the color map
    sm = plt.cm.ScalarMappable(cmap='viridis_r')


    image = ax.imshow_hpx((hpx, 'ICRS'), cmap='viridis_r', zorder=9, alpha=.6)

    cbar = plt.colorbar(sm)
    vmin=minval
    vmax=max(hpx)

    sm.set_norm(LogNorm(vmin=vmin, vmax=vmax))

    cbar.set_label('Flux upper limit (erg cm$^{-2}$ s$^{-1}$)', fontsize=14)

    if skymap_file!='None' and skymap_file[0]!='point':
        map_name=extract_final_part(skymap_file)
        plt.title('event = %s, skymap = %s, \n Temporal bin = 1.024 s, spectrum : normal ' %(name,map_name), fontsize=18)
    
    else:
        plt.title('Temporal bin = 1.024 s, spectrum : normal ' , fontsize=18)


    plt.savefig(os.path.join(path_results,'ul.pdf'))


def ul_gw(gwid,t0,head,ras,decs,wdir,path_results, ratio, ul_arr):

    ul_final={}

    client = GraceDb()

    gw=client.files('%s' %gwid).json().keys()

    logging.info('check gw ul')

    gw=list(gw)
    check=0

    for i in range(len(gw)):

        if 'Bilby.multiorder.fits'==gw[i]:
            check=1
            graceid=gwid
            map_name=gw[i]
            down(graceid,map_name,wdir)
            break

        if gw[i].endswith('multiorder.fits'):
            check=1
            graceid=gwid
            map_name=gw[i]
            down(graceid,map_name,wdir)  
            break

    if check==0:
        logging.info('No skymaps for %s' %gw[i])
    else:
        logging.info('%s done' %gwid)


    skymap_file=os.path.join(wdir,map_name)
    skymap = Table.read(skymap_file)

    # this part is needed to not consider part of sky behind the earth

    logging.info('check pre earth')


    while True:
        try:
            ra_ea, red_ea, radius_ear = get_earth_sat_pos(t0, orbitdat=None)
            break
        except:
            logging.info('error in getting earth data')

    logging.info('check post earth')


    # This part avoids looping 4 times in the conversion i-pixel to ra-dec
    ra_map=np.zeros(len(skymap['PROBDENSITY']))
    dec_map=np.zeros(len(skymap['PROBDENSITY']))
    nside_map=np.zeros(len(skymap['PROBDENSITY']))

    for i in range(len(skymap['PROBDENSITY'])):


        uniq = skymap[i]['UNIQ']
        level, ipix = ah.uniq_to_level_ipix(uniq)
        nside = ah.level_to_nside(level)

        nside_map[i]=nside
        ra, dec = ah.healpix_to_lonlat(ipix, nside, order='nested')
        ra_map[i] = ra.deg
        dec_map[i] = dec.deg


    # print(nside_map)

    cumul=0
    tot=0
    for i in range(len(skymap['PROBDENSITY'])):

        # logging.info(i)

        ra, dec = ra_map[i], dec_map[i]
        nside = nside_map[i]

        tot+=skymap[i]['PROBDENSITY'] * (np.pi / 180)**2*hp.nside2pixarea(nside, degrees=True)

        if ang_sep(ra_ea, red_ea, ra, dec)< radius_ear:

            cumul+=skymap[i]['PROBDENSITY'] * (np.pi / 180)**2*hp.nside2pixarea(nside, degrees=True)

            skymap[i]['PROBDENSITY']=0

    # this line prints the fraction of the gw posterior occulted by earth
    sentence = 'The fraction of the sky posterior occulted by Earth is %%%% %'
    # Format the sentence with the number
    formatted_sentence = sentence.replace('%%%%', "%.2f" % (cumul/tot*100))

    # Save the sentence to a text file
    filename = os.path.join(path_results,'earth_occultation.txt')  # Specify the desired filename
    with open(filename, 'w') as file:
        file.write(formatted_sentence)


   
    df = pd.DataFrame(columns=['soft', 'normal', 'hard', '170817'], index=['0.128', '0.256', '0.512', '1.024', '2.048', '4.096','8.192', '16.384'])

    logging.info('check pre ul')


    for n in range(len(head)):
        logging.info('loop gw')

       
        spec = head[n][0]
        time_bin = head[n][1]

        if time_bin=='0.128':

            ul_5sigma_new = ul_arr['%s' %spec]


            def f_ul_n(x,y):
                return f_ul(x,y,ras,decs,ul_5sigma_new)

            dp=0
            norm=0
            for i in range(len(skymap['PROBDENSITY'])):


                ra, dec = ra_map[i], dec_map[i]
                nside = nside_map[i]    
                ul=f_ul_n(ra,dec)



                if skymap[i]['PROBDENSITY']>0:

                    dp+=ul*skymap[i]['PROBDENSITY'] * (np.pi / 180)**2*hp.nside2pixarea(nside, degrees=True)
                    norm+=skymap[i]['PROBDENSITY'] * (np.pi / 180)**2*hp.nside2pixarea(nside, degrees=True)


            df.at[time_bin,spec] = "{:.2e}".format(dp/norm)

            ul_final['%s' %spec]=dp/norm

        else:
             df.at[time_bin,spec] = "{:.2e}".format(ul_final['%s' %spec]*ratio['%s' %spec]['%s' %time_bin])
           

        if spec=='normal' and time_bin=='1.024':
            ul_5sigma_new= np.array(ul_arr['%s' %spec]) * ratio['%s' %spec]['%s' %time_bin]
            plot_ul_map(ras,decs,ul_5sigma_new, skymap_file , ra_ea, red_ea, radius_ear, gwid, path_results)


    logging.info('check post ul')

    df.to_csv(os.path.join(path_results,'ul_gw.txt'), sep='\t', index=True)

    


def ul_nogw(t0,head,ras,decs,wdir, path_results, ratio, ul_arr, coord_file):

    fermi=False

    ul_final={}


    logging.info('start nogw')

    region=None


    if coord_file != None:

        filename = os.path.join(wdir,coord_file)

        with open(filename, 'r') as file:
            # Read the first line to get the requested variable
            row = file.readline().strip()

            if row=='circle':
                region='circle'
                values = next(file).strip().split(',')
                ra_circ = float(values[0])
                dec_circ = float(values[1])
                radius_circ = float(values[2])
            
            elif row=='poly':
                region='poly'
                data = np.loadtxt(file, delimiter=',')
                ra_poly = data[:, 0]
                dec_poly = data[:, 1]

            elif row=='point':
                region='point'
                values = next(file).strip().split(',')
                ra_point = float(values[0])
                dec_point = float(values[1])

            else:
                logging.info('no valid file')

        if region=='poly':
            ra_pts=ra_poly.tolist()
            dec_pts=dec_poly.tolist()
            verts_map = HealPix.from_vertices(ra_pts, dec_pts, nside=128)

            def weight(ra,dec):
                return verts_map.probability(ra, dec)
        
        if region=='circle':

            def weight(ra,dec):

                if ang_sep(ra_circ, dec_circ, ra, dec)< radius_circ:
                    return 1.0
                else:
                    return 0.0

    else:

        trigcat = TriggerCatalog()


        format = '%Y-%m-%d %H:%M:%S.%f'

        dt = datetime.strptime(t0, format)

        t1 = dt - timedelta(seconds=1)
        t2 = dt + timedelta(seconds=1)

        t1=t1.strftime(format)
        t2=t2.strftime(format)

        tri_name = trigcat.slice('trigger_time', lo=t1, hi= t2).get_table(columns=('trigger_name', 'trigger_time'))

        if len(tri_name)>0:
        
            tri_name= tri_name[0][0]

            os.system('curl -o %s/glg_healpix_all_%s.fit https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/20%s/%s/quicklook/glg_healpix_all_%s.fit' %(wdir,tri_name,tri_name[2:4],tri_name,tri_name))

            name_fermi_map='glg_healpix_all_%s.fit' %tri_name

            if os.path.exists(os.path.join(wdir,name_fermi_map)):

                fermi=True

                skymap_file=os.path.join(wdir,name_fermi_map)

                loc = GbmHealPix.open(os.path.join(wdir,name_fermi_map))

                logging.info('fermi map received')
                def weight(ra,dec):
                    return loc.probability(ra, dec)
                
            else:
                logging.info('no Fermi skymap available')

                def weight(ra,dec):
                    return 1.0

        else:

            def weight(ra,dec):
                return 1.0



    #logging.info('check pre earth')


    while True:
        try:
            ra_ea, red_ea, radius_ear = get_earth_sat_pos(t0, orbitdat=None)
            logging.info('earth map received')
            break
        except:
            logging.info('error in getting earth data')

    
    if region=='point':

        df = pd.DataFrame(columns=['soft', 'normal', 'hard', '170817'], index=['0.128', '0.256', '0.512', '1.024', '2.048', '4.096','8.192', '16.384'])

        for n in range(len(head)):

        
            spec = head[n][0]
            time_bin = head[n][1]

            if time_bin=='0.128':

                ul_5sigma_new = ul_arr['%s' %spec]


                def f_ul_n(x,y):
                    return f_ul(x,y,ras,decs,ul_5sigma_new)

                ul=f_ul_n(ra_point,dec_point)

                df.at[time_bin,spec] = "{:.2e}".format(ul)

                ul_final['%s' %spec]=ul
            
            else:
             
                df.at[time_bin,spec] = "{:.2e}".format(ul_final['%s' %spec]*ratio['%s' %spec]['%s' %time_bin])

            if spec=='normal' and time_bin=='1.024':
                ul_5sigma_new= np.array(ul_arr['%s' %spec]) * ratio['%s' %spec]['%s' %time_bin]
                skymap_file=['point',ra_point,dec_point]
                plot_ul_map(ras,decs,ul_5sigma_new, skymap_file, ra_ea, red_ea, radius_ear, None, path_results)


        df.to_csv(os.path.join(path_results,'ul.txt'), sep='\t', index=True)
        
    else:
        #logging.info('check post earth')
    
        nside = 32  # Resolution parameter, determines the number of pixels (higher resolution = more pixels)
        npix = hp.nside2npix(nside)  # Total number of pixels in the map
    
        map_values = np.zeros(npix)  # Initialize the array with zeros
    
        tot=0
        cumul=0
    
        for n in range(npix):
    
            theta, phi = hp.pix2ang(nside, n)  # Retrieve theta and phi angles
            dec = np.degrees(np.pi / 2 - theta)  # Convert theta to Dec
            ra = np.degrees(phi)  # Convert phi to RA
    
    
            tot+=weight(ra,dec)
    
            if ang_sep(ra_ea, red_ea, ra, dec)< radius_ear:
                map_values[n]=0.0
                cumul+=weight(ra,dec)
    
            else:
                map_values[n]=1.0 * weight(ra,dec)

        logging.info('earth fraction')

        # this line prints the fraction of the gw posterior occulted by earth
        sentence = 'The fraction of the sky posterior occulted by Earth is %%%% %'
        # Format the sentence with the number
        formatted_sentence = sentence.replace('%%%%', "%.2f" %(cumul/tot*100))
    
        # Save the sentence to a text file
        filename = os.path.join(path_results,'earth_occultation.txt')  # Specify the desired filename
        with open(filename, 'w') as file:
            file.write(formatted_sentence)
    
        if max(map_values)==0.0:
            logging.info('region occulted by earth, no upper limits available')
        
        else:
    
            #logging.info('check pre ul')
    
            df = pd.DataFrame(columns=['soft', 'normal', 'hard', '170817'], index=['0.128', '0.256', '0.512', '1.024', '2.048', '4.096','8.192', '16.384'])
    
            for n in range(len(head)):

    
            
                spec = head[n][0]
                time_bin = head[n][1]
                
                logging.info('ul weighting %s %s' %(spec, time_bin))
  

                if time_bin=='0.128':

                    ul_5sigma_new = ul_arr['%s' %spec]


                    def f_ul_n(x,y):
                        return f_ul(x,y,ras,decs,ul_5sigma_new)
        
                    dp=0
                    norm=0
        
                    for i in range(npix):
        
                        theta, phi = hp.pix2ang(nside, i)  # Retrieve theta and phi angles
                        dec = np.degrees(np.pi / 2 - theta)  # Convert theta to Dec
                        ra = np.degrees(phi)   # Convert phi to RA
        
        
                        if map_values[i]>0:
        
                            ul=f_ul_n(ra,dec)
                            dp+=ul*map_values[i] 
                            norm+=map_values[i]


                    df.at[time_bin,spec] = "{:.2e}".format(dp/norm)

                    ul_final['%s' %spec]=dp/norm
                
                else:

                    df.at[time_bin,spec] = "{:.2e}".format(ul_final['%s' %spec]*ratio['%s' %spec]['%s' %time_bin])

    
                if spec=='normal' and time_bin=='1.024':

                    ul_5sigma_new= np.array(ul_arr['%s' %spec]) * ratio['%s' %spec]['%s' %time_bin]

                    if fermi:
                        plot_ul_map(ras,decs,ul_5sigma_new, skymap_file , ra_ea, red_ea, radius_ear, tri_name, path_results)
                    else:
                        plot_ul_map(ras,decs,ul_5sigma_new,'None', ra_ea, red_ea, radius_ear, tri_name, path_results)

    
                
    
            #logging.info('check post ul')

            df.to_csv(os.path.join(path_results,'ul.txt'), sep='\t', index=True)

            logging.info('ul done')

def  compute_ul(trig_time,head,ras,decs,work_dir, path_results, ratio, ul_arr, coord_file):

    logging.basicConfig(
        filename=os.path.join(path_results,'upper_limits_analysis.log'),
        level=logging.DEBUG,
        format="%(asctime)s-" "%(levelname)s- %(message)s",
    )


    utc_time = Time(trig_time, scale='utc')
    t1=utc_time.gps-1
    t2=utc_time.gps+1

    client = GraceDb()

    ev=client.superevents(query='gpstime: %s .. %s' %(str(t1),str(t2)))

    ev_list=[]
    for event in ev:
        ev_list.append(event['superevent_id'])

    #logging.info('ev list', ev_list)


    try:
        trig_time = datetime.strptime(trig_time, '%Y-%m-%dT%H:%M:%S.%f').strftime('%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        trig_time = datetime.strptime(trig_time, '%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')

    


    if len(ev_list)>0:
        ul_gw(ev_list[0],trig_time,head,ras,decs,work_dir, path_results, ratio, ul_arr)
    else:
        
        ul_nogw(trig_time,head,ras,decs,work_dir, path_results, ratio, ul_arr, coord_file)


