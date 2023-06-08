import numpy as np

import healpy as hp

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm

import os
import re

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as io


# make theta vs. phi plot of DeltaLLH
def theta_phi_dlogl_plot(res_hpmax_tab, norm=None, vmin=None, vmax=None, theta=None, phi=None):
    fig=plt.figure(figsize=(12,5))
    if norm is None:
        plt_thing = (2*(res_hpmax_tab['nllh'] - np.min(res_hpmax_tab['nllh']\
                                      [np.isfinite(res_hpmax_tab['nllh'])])))
        plt.scatter(res_hpmax_tab['phi'], res_hpmax_tab['theta'], c=plt_thing,\
                    norm=LogNorm(), cmap=cm.Spectral, s=6, vmin=vmin, vmax=vmax)
        plt.colorbar(label=r'2$\Delta$LLH')
    elif norm == 'log':
        plt_thing = 1+(2*(res_hpmax_tab['nllh'] - np.min(res_hpmax_tab['nllh']\
                                      [np.isfinite(res_hpmax_tab['nllh'])])))
        plt.scatter(res_hpmax_tab['phi'], res_hpmax_tab['theta'], c=plt_thing,\
                    norm=LogNorm(), cmap=cm.Spectral, s=6, vmin=vmin, vmax=vmax)
        plt.colorbar(label=r'2$\Delta$LLH + 1')
    elif norm == 'sqrt':
        plt_thing = np.sqrt(2*(res_hpmax_tab['nllh'] - np.min(res_hpmax_tab['nllh']\
                                      [np.isfinite(res_hpmax_tab['nllh'])])))
        plt.scatter(res_hpmax_tab['phi'], res_hpmax_tab['theta'], c=plt_thing,\
                    norm=LogNorm(), cmap=cm.Spectral, s=6, vmin=vmin, vmax=vmax)
        plt.colorbar(label=r'$\sqrt{2\Delta LLH}$')
    if phi is not None and theta is not None:
        # optional point to plot, like an external best fit position
        plt.plot(phi, theta, '*', markersize=4)
    plt.grid(True)
    plt.xlabel('Phi')
    plt.ylabel('Theta')
    return fig


# make healpix map of DeltaLLH
def mk_hp_map(res_hpmax_tab):
    Nside = 2**4
    dlogl_map = hp.UNSEEN*np.ones(hp.nside2npix(Nside))
    dlogl_map[res_hpmax_tab['hp_ind']] = (res_hpmax_tab['nllh'] - np.min(res_hpmax_tab['nllh']\
                                                              [np.isfinite(res_hpmax_tab['nllh'])]))
    return dlogl_map


def mk_mhp_map(res_hpmax_tab):
    from mhealpy import HealpixMap

    m = HealpixMap(nside = 2**4, scheme ='NESTED', dtype = float)
    m[res_hpmax_tab['hp_ind']] = (res_hpmax_tab['nllh'] - np.min(res_hpmax_tab['nllh']\
                                                              [np.isfinite(res_hpmax_tab['nllh'])]))
    return m


def display_hp_map(res_hpmax_tab, norm=None, vmin=None, vmax=None, ra=None, dec=None):
    fig=plt.figure(figsize=(10,10))

    dlogl_map = mk_hp_map(res_hpmax_tab)
    if norm is None:
        hp.mollview(dlogl_map, nest=True, max=vmax, min=vmin, rot=(180.0,0,0))
    else:
        # add 1 to not throw and error for a log norm
        hp.mollview(1+dlogl_map, nest=True, max=vmax, min=vmin, norm=norm, rot=(180.0,0,0))
    if ra is not None and dec is not None:
        # optional point to plot, like an external best fit position
        hp.projscatter(ra, dec, lonlat=True)
    hp.graticule()


def plotly_dlogl_sky(trigid, res_out_tab, config_id=0, outdir=''):
    # find timeID with max TS
    maxTIMEid = res_out_tab['timeID'][res_out_tab['TS'].argmax()]

    # for the results for the maxTIMEid, find the max TS results for each position/hp index 
    bl = np.isclose(res_out_tab['timeID'], maxTIMEid)
    idx = res_out_tab[bl].groupby(['hp_ind'])['TS'].transform(max) == res_out_tab[bl]['TS']
    res_hpmax_tab = res_out_tab[bl][idx]
    ras,decs=hp.pix2ang(2**4,res_hpmax_tab['hp_ind'].values,lonlat=True,nest=True)
    plt_thing_unlog = ((2*(res_hpmax_tab['nllh'] - np.min(res_hpmax_tab['nllh']\
                                        [np.isfinite(res_hpmax_tab['nllh'])]))))
    plt_thing=np.where(plt_thing_unlog != 0,  np.log10(plt_thing_unlog), 0)

    special = np.empty(shape=(len(res_hpmax_tab),3,1), dtype='object')
    try:
        special[:, 0] = np.array(res_hpmax_tab['theta']).reshape(-1,1)
        special[:, 1] = np.array(res_hpmax_tab['phi']).reshape(-1,1)
        special[:, 2] = np.array(plt_thing_unlog).reshape(-1,1)
    except:
        print('there')

    fig=px.scatter_geo(lon=ras,lat=decs, color=plt_thing,
                       color_continuous_scale=px.colors.diverging.Spectral,
                       labels={
                           "lon": "RA (deg)",
                           "lat": "Dec (deg)",
                           "color": "&#916;LLH"
                       },
                       title=f'&#916;LLH for OFOV Positions, Config {config_id}'
                    )
    fig.update_traces(
                    customdata=special,
                    hovertemplate=
                    '<i>RA</i>: %{lon:.2f} ' +
                    '<i>Dec</i>: %{lat:.2f}<br>' +
                    '<i>Theta</i>: %{customdata[0]:.2f} ' +
                    '<i>Phi</i>: %{customdata[1]:.2f}<br>' +
                    '<b>2&#916;LLH: %{customdata[2]:.2f}</b>'
                    )

    fig.update_geos(projection_type="mollweide",
                    showcoastlines=False, showland=False,
                    lataxis_showgrid=True, lonaxis_showgrid=True,
                    )

    fig.update_layout(coloraxis_colorbar=dict(
        title='log<sub>10</sub>2&#916;LLH',
        titleside='top',
        ticktext=plt_thing_unlog
    ))

    data = fig
    filename = os.path.join(outdir, f'{trigid}_{config_id}_n_OUTFOV.json')

    with open(filename, 'w') as f:
        f.write(io.to_json(data))
    return filename


def plotly_waterfall_seeds(rates, trigid, config_id=0, outdir=''):

    magma_cmap = cm.get_cmap('magma')
    norm = matplotlib.colors.Normalize(0,max(rates.snr))

    fig1=go.Figure()
    for index, row in rates.iterrows():
        snr=row['snr']
        color='rgb'+str(matplotlib.colors.colorConverter.to_rgb(magma_cmap(norm(row['snr']))))
        fig1.add_trace(go.Scatter(
                x=[row['dt'], row['dt']+row['duration'], row['dt']+row['duration'],row['dt'],row['dt']],
                y=[row['duration'],row['duration'],row['duration']*2,row['duration']*2,row['duration']],
                mode='lines',
                line=dict(width=0),
                fill='toself',
                fillcolor=color,
                opacity=0.6,
                text=round(row['snr'],2),
                hoverinfo='text'
            ))

    fig1.update_yaxes(type='log', dtick=0.30102999566,range=[-1,1.5])
    fig1.update_xaxes(range=[-20,20])

    colorbar_trace = go.Scatter(x=[None],
                                 y=[None],
                                 mode='markers',
                                 marker=dict(
                                     colorscale='Magma', 
                                     showscale=True,
                                     cmin=0,
                                     cmax=max(rates.snr),
                                     colorbar=dict(thickness=10,
                                                   outlinewidth=0,
                                                    title ='SNR')
                                 ),
                                 hoverinfo='none'
                                )
    fig1.update_layout(xaxis_title='T-T0 (s)', yaxis_title='Timescale (s)', 
                       title=f'Full Rates SNR per Time Bin, Config {config_id}')
    fig1['layout']['showlegend'] = False
    fig1.add_trace(colorbar_trace)
    data=fig1
    filename = os.path.join(outdir, f'{trigid}_{config_id}_n_FULLRATE.json')

    with open(filename, 'w') as f:
        f.write(io.to_json(data))
    return filename


def plotly_splitrates(trigid, splitrates, config_id=0, outdir=''):
    maxTIMEid = splitrates['timeID'].iloc[splitrates['TS'].argmax()]
    maxTIMEonly = splitrates.loc[splitrates['timeID']==maxTIMEid]

    ifovonly = maxTIMEonly
    
    delllh = -(ifovonly['TS'] - np.max(ifovonly['TS']\
                                        [np.isfinite(ifovonly['TS'])]))
    sqrtllh = np.sqrt(delllh.values)
    
    ras = []
    decs = []
    for index, row in ifovonly.iterrows():
        arr = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", row['ra_dec'])
        ra = float(arr[0])
        dec = float(arr[1])
        ras.append(ra)
        decs.append(dec)

    fig = px.scatter_geo(lon=ras,lat=decs,color=sqrtllh,
                        labels={
                            "x": "IMX",
                            "y": "IMY",
                            "color": "&#x221A;&#916;&#923;"
                        },
                        title = f"Split-Rate Results for Best Time Bin ({ifovonly.dur.iloc[0]}s at dt = {round(maxTIMEonly.dt.iloc[0],2)})), Config {config_id}")

    special = np.empty(shape=(len(ifovonly), 3, 1), dtype='object')
    special[:, 0] = np.array(ifovonly.theta).reshape(-1, 1)
    special[:, 1] = np.array(ifovonly.phi).reshape(-1, 1)
    special[:, 2] = np.array(sqrtllh).reshape(-1, 1)
    fig.update_traces(
                    customdata=special,
                    hovertemplate =
                    '<i>RA,Dec</i>: %{lon:.2f}, %{lat:.2f}<br>' +
                    '<i>Theta,Phi</i>: %{customdata[0]:.2f}, %{customdata[1]:.2f}<br> ' +
                    '<b>&#x221A;&#916;&#923;: %{customdata[2]:.2f}</b>'
                    )
    
    fig.update_geos(projection_type="mollweide",
                    showcoastlines=False,
                    showland=False,
                    lataxis_showgrid=True,
                    lonaxis_showgrid=True)
    data = fig
    filename = os.path.join(outdir, f'{trigid}_{config_id}_n_SPLITRATE.json')

    with open(filename, 'w') as f:
        f.write(io.to_json(data))
    return filename
