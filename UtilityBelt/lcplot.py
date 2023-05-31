from .functions import get_64ms_rates, get_swift_attitude
from .functions import get_quad_rates


from swifttools.swift_too import Clock
import matplotlib.pyplot as plt
import numpy as np
import os

import plotly.graph_objects as go
import plotly.io as io


def plot_64ms(trigid, trigtime, outdir='', duration=100):
    fig = plt.figure(figsize=(5, 5), dpi=125)

    att_data, utcf = get_swift_attitude(trigtime)
    trigger_time = Clock(utctime=trigtime).met
    att_ind = np.argmin(np.abs(att_data['TIME'] - trigger_time))
    pnt_ra, pnt_dec = att_data['POINTING'][att_ind,:2]

    plt.plot(att_data['TIME']-trigger_time, att_data['POINTING'], '.')
    plt.title(f'{round(pnt_ra,3), round(pnt_dec,3)} at T0')
    plt.grid(True)
    plt.xlim(-duration/2, duration/2)

    plt.legend(['RA','Dec','Roll'])
    plt.xlabel('T - T0 (s)')
    plt.ylabel('RA/Dec/Roll (deg)')

    filename = os.path.join(outdir, f'{trigid}_ATTPLOT.png')
    plt.savefig(filename)
    
    plt.close(fig)

    return filename


def plotly_64ms(trigtime, duration=100):
    ratedata, utcf = get_64ms_rates(trigtime)
    trigger_time = Clock(utctime=trigtime).met

    ratedat = ratedata[(ratedata['TIME']-trigger_time < duration) & (ratedata['TIME']-trigger_time > -duration)]

    if len(ratedat) == 0:
        raise ValueError(f"No data in +- {duration} window.")

    ebins=["15-25", "25-50", "50-100", "100-350"]
    ind = 0
    traces=[]
    for ebin in ebins:
        trace1 = go.Scatter(x=ratedat['TIME']-trigger_time,
                            y=ratedat['COUNTS'][:, ind]*15.625,
                            mode='lines',
                            line_shape='hv',
                            name=ebin)
        ind += 1
        traces.append(trace1)

    trace1 = go.Scatter(x=ratedat['TIME']-trigger_time,
                        y=np.sum(ratedat['COUNTS'], axis=1)*15.625,
                        mode='lines',
                        line_shape='hv',
                        name='15-350')
    traces.append(trace1)

    fig = go.Figure(data=traces)
    fig.update_xaxes(range=[-duration, duration])
    fig.update_layout(
        title="64 ms Rates",
        xaxis_title="T-T0 (s)",
        yaxis_title="Counts/s",
        legend_title="Energy Bin (keV)",
        )

    return fig


def plotly_quads(trigtime, duration=100):
    ratedata, utcf = get_quad_rates(trigtime)
    trigger_time= Clock(utctime=trigtime).met

    ratedat=ratedata[(ratedata['TIME']-trigger_time < duration) & (ratedata['TIME']-trigger_time > -duration)]

    if len(ratedat) == 0:
        raise ValueError(f"No data in +- {duration} window.")

    ebins=["15-25","25-50","50-100","100-350"]
    ind = 0
    traces=[]
    erates=[]
    for ebin in ebins:
        erate=ratedat['QUAD0_COUNTS'][:,ind]+ratedat['QUAD1_COUNTS'][:,ind]+ratedat['QUAD2_COUNTS'][:,ind]+ratedat['QUAD3_COUNTS'][:,ind]
        erates.append(erate)
        trace1 = go.Scatter(x=ratedat['TIME']-trigger_time,
                            y=erate/1.6,
                            mode='lines',
                            line_shape='hv',
                            name=ebin)
        ind += 1
        traces.append(trace1)

    trace1 = go.Scatter(x=ratedat['TIME']-trigger_time,
                        y=np.sum(erates, axis=0)/1.6,
                        mode='lines',
                        line_shape='hv',
                        name='15-350')
    traces.append(trace1)

    fig = go.Figure(data=traces)
    fig.update_xaxes(range=[-duration, duration])
    fig.update_layout(
        title="1.6 s Rates",
        xaxis_title="T-T0 (s)",
        yaxis_title="Counts/s",
        legend_title="Energy Bin (keV)",
        )

    return fig


def plotly_rates(trigid, trigtime, outdir=''):
    data1 = io.to_json(plotly_64ms(trigtime))
    data2 = io.to_json(plotly_quads(trigtime))
    data = '['+data1+','+data2+']'  # stupid shit
    filename = os.path.join(outdir, f'{trigid}_LIGHTCURVE.json')
    with open(filename, 'w') as f:
        f.write(data)
    return filename
