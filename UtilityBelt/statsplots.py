import plotly
import plotly.express as px
import numpy as np
import json
import pandas as pd


def TShist(df):
    df['duration_rounded'] = np.round(df['duration'],3)

    df['log_maxTS'] = np.log10(df['maxTS'])

    # Create the histogram
    fig = px.histogram(df, x='log_maxTS', color='duration_rounded', nbins=200, hover_data=df.columns)

    # Update x-axis title and tickvals/ticktext to show original values instead of log-transformed
    xaxis_title = 'Maximum Test Statistic per trigger'
    fig.update_xaxes(title_text=xaxis_title, tickvals=[np.log10(x) for x in [1,10,100,1000,10000]], ticktext=['1','10','100','1000','10000'])
    fig.update_yaxes(title_text="Count",type="log")
    fig.update_layout(legend_title_text='Duration')
    fig.add_vline(x=np.log10(8))

    data = fig
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    
    return graphJSON

def inv_symlog(arr, base=10, linthresh=2, linscale=1):
    _linscale_adj = (linscale / (1.0 - base ** -1))
    _log_base = np.log(base)

    arr = np.asarray(arr)  # convert to numpy array
    abs_arr = np.abs(arr)

    # arrays with all elements within the linear region of this inverse symlog transformation 
    linear = np.max(abs_arr, axis=0) < _linscale_adj if arr.size > 1 else abs_arr < _linscale_adj

    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.sign(arr) * linthresh * (np.power(base, abs_arr / linthresh - _linscale_adj))
        inside = abs_arr <= _linscale_adj if arr.size > 1 else abs_arr <= _linscale_adj

    if arr.size > 1:
        out[inside] = arr[inside] / _linscale_adj
    else:
        out = out / _linscale_adj if inside else out

    out = out/np.max(np.abs(out), axis=0)*np.log(np.max(abs_arr, axis=0))/_log_base if arr.size > 1 else out/np.abs(out)*np.log(abs_arr)/_log_base

    if arr.size > 1:
        out[np.array([linear]*out.shape[0])] = arr[np.array([linear]*out.shape[0])]
    else:
        out = arr if linear else out

    return out

def symlog(arr, base=10, linthresh=2, linscale=1):
    _linscale_adj = (linscale / (1.0 - base ** -1))
    _log_base = np.log(base)
    arr = np.array(arr)
    abs_arr = np.abs(arr)
    
    # arrays with all elements within the linear region of this symlog transformation 
    linear = np.max(abs_arr, axis=0) < linthresh

    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.sign(arr) * linthresh * (
            _linscale_adj +
            np.log(abs_arr / linthresh) / _log_base)
        inside = abs_arr <= linthresh

    out[inside] = arr[inside] * _linscale_adj

    out = out/np.max(np.abs(out), axis=0)*np.log(np.max(abs_arr, axis=0))/_log_base

    out[np.array([linear]*out.shape[0])] = arr[np.array([linear]*out.shape[0])]
    
    return out

def peakvout(df):
    df['log_maxTS'] = np.log10(df['maxTS'])
    df['DeltaLLHOut'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['DeltaLLHOut'], inplace=True)

    df['DeltaLLHPeak'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['DeltaLLHPeak'], inplace=True)

    df['symlog_DeltaLLHOut'] = symlog(df['DeltaLLHOut'])

    # In your scatter plot, add this new column to the hover data
    fig = px.scatter(df, 
                    x='DeltaLLHPeak', 
                    y='symlog_DeltaLLHOut', 
                    color='log_maxTS', 
                    color_continuous_scale='Plasma', 
                    log_x=True, 
                    hover_data={
                        'trigger_id': True, 
                        'DeltaLLHPeak': True, 
                        'DeltaLLHOut': True, 
                        'maxTS': True,
                        'log_maxTS': False,
                        'symlog_DeltaLLHOut': False
                    })

    # Set the colorbar title and tickvals
    colorbar_title = 'Max TS'
    colorbar_tickvals = [0, 1, 2, 3]  # adjust these to match the range of values in 'maxTS'
    colorbar_ticktext = [10**val for val in colorbar_tickvals]

    min_log_y = np.floor(np.log10(np.abs(df['DeltaLLHOut']).min()))
    max_log_y = np.ceil(np.log10(np.abs(df['DeltaLLHOut']).max()))

    # Create positive and negative ticks
    positive_yticks_unsymlog = np.logspace(0, max_log_y, num=int(max_log_y)+1)
    negative_yticks_unsymlog = -np.logspace(0, abs(min_log_y), num=int(abs(min_log_y))+1)

    # Concatenate negative and positive ticks
    yticks_unsymlog = np.concatenate((negative_yticks_unsymlog, positive_yticks_unsymlog))

    # Apply the symlog transformation to the tick values
    yticks_symlog = symlog(yticks_unsymlog)

    # Update the y-axis tick values and labels
    fig.update_yaxes(tickvals=yticks_symlog, ticktext=yticks_unsymlog, title_text='DeltaLLHOut')

    fig.update_layout(coloraxis_colorbar=dict(
        title=colorbar_title,
        tickvals=colorbar_tickvals,
        ticktext=colorbar_ticktext))

    fig.add_vline(x=10)
    fig.add_hline(y=1.6)
    data = fig
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON