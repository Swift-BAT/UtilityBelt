from functions import get_swift_attitude

from swifttools.swift_too import Clock
import matplotlib.pyplot as plt
import numpy as np
import os

import plotly.graph_objects as go
import plotly.io as io


def attitude_plot(trigid, trigtime, outdir='', duration=300):
    fig = plt.figure(figsize=(5, 5), dpi=125)

    att_data, utcf = get_swift_attitude(trigtime)
    trigger_time= Clock(utctime=trigtime).met
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


def plotly_attitude(trigid, trigtime, outdir='', duration=100):

    satdata, utcf = get_swift_attitude(trigtime)
    trigger_time= Clock(utctime=trigtime).met

    att_data=satdata[(satdata['TIME']-trigger_time < duration) & (satdata['TIME']-trigger_time > -duration)]

    att_ind = np.argmin(np.abs(att_data['TIME'] - trigger_time))

    angles=['RA','Dec','Roll']
    ind=0
    traces=[]
    for angle in angles:
        trace1 = go.Scatter(x=att_data['TIME']-trigger_time,
                            y=att_data['POINTING'][:, ind],
                            name=angle,mode='markers',
                            marker_symbol='circle')
        ind += 1
        traces.append(trace1)

    fig = go.Figure(data=traces)
    fig.update_xaxes(range=[-duration, duration])
    fig.update_layout(
        title="Attitude Data",
        xaxis_title="T-T0 (s)",
        yaxis_title="Degrees",
        )
    
    data = fig

    filename = os.path.join(outdir, f'{trigid}_ATTITUDE.json')

    with open(filename, 'w') as f:
        f.write(io.to_json(data))
    return filename
