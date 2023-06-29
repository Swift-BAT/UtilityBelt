import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import plotly.graph_objs as go
import plotly.io as io
import os


def dpiplot(eventfile, trigid, outdir=''):
    ev_data=eventfile[1].data
    exposure=eventfile[1].header['EXPOSURE']
    
    plt.figure(figsize=(15,10))
    xbins = np.arange(286+1) - 0.5
    ybins = np.arange(173+1) - 0.5
    plt.hist2d(ev_data['DETX'], ev_data['DETY'], bins=[xbins,ybins], norm=colors.LogNorm())
    plt.xlabel("DETX", size=15)
    plt.ylabel("DETY", size=15)
    plt.colorbar(label="Number of Counts")
    plt.title(f"{trigid} {round(exposure)}s {len(set(ev_data.DET_ID))} active dets")

    filename = os.path.join(outdir, f'{trigid}_DPI.png')
    plt.savefig(filename)
    
    plt.close(fig)

    return filename

    
def dpiplotly(eventfile, trigid, outdir=''):
    ev_data=eventfile[1].data
    exposure=eventfile[1].header['EXPOSURE']
    
    DETX_bins = np.histogram_bin_edges(ev_data['DETX'], bins=286)
    DETY_bins = np.histogram_bin_edges(ev_data['DETY'], bins=173)

    heatmap = go.Histogram2d(
        x=ev_data['DETX'],
        y=ev_data['DETY'],
        xbins=dict(start=DETX_bins[0], end=DETX_bins[-1], size=(DETX_bins[-1] - DETX_bins[0])/len(DETX_bins)),
        ybins=dict(start=DETY_bins[0], end=DETY_bins[-1], size=(DETY_bins[-1] - DETY_bins[0])/len(DETY_bins)),
        colorscale='Jet',  # or any other colorscale
        zmax=np.log(ev_data['DETX'].max()),  # log scale
        zmin=np.log(ev_data['DETY'].min()),
        hovertemplate = '<i>DETX</i>: %{x}' + '<br><i>DETY</i>: %{y}<br><i>Counts</i>: %{z}<extra></extra>',
    )

    layout = go.Layout(
        title=dict(text=f"{trigid} {round(exposure)}s {len(set(ev_data.DET_ID))} active dets"),
        xaxis=dict(title='DETX'),
        yaxis=dict(title='DETY'),
        autosize=False,
        width=990,
        height=600,
        coloraxis_colorbar=dict(title="Number of Counts"),
    )

    fig = go.Figure(data=[heatmap], layout=layout)
    data = fig
    filename = os.path.join(outdir, f'{trigid}_DPI.json')

    with open(filename, 'w') as f:
        f.write(io.to_json(data))
    return filename