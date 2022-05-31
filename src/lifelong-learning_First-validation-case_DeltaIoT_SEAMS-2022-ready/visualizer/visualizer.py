import plotly.graph_objects as go
import plotly.express as px
import random
from common.stream import *
from common.goal import *
import pickle
from tqdm import tqdm
import numpy as np

def compute_metrics(target_type, metric,stream, base_addr = "./data/predicts/"):
    stream.read_current_cycle()

    metric_vals = []
    for i in tqdm(range(1, stream.cycles_num)):
        # reading actual values
        features, targets = stream.read_current_cycle()
        targets = targets[target_type]

        #reading predicted values
        predicts, selected_option_id = pickle.load(open(base_addr +\
                               str(target_type) + \
                               "/cycle" + str(stream.current_cyle - 1) + ".pkl", "rb"))


        metric_vals.append(metric(predicts[target_type], targets))
    return metric_vals

def plot_metrics(metrics, plot_names, cycle_num, cut_x_ranges, 
                 cut_y_ranges, metric_name, shape_name, 
                 text_title, y_range = None, add_vertical = True, line_colors = None):

    plots = []
    max_val = np.amax(metrics)
    for i in range(len(metrics)):
        if(line_colors is not None):
            plots.append(go.Scatter(x = list(range(2, cycle_num + 1)), y = metrics[i], name = plot_names[i],  
                                line=dict(color=line_colors[i]), line_shape="spline", line_width=2))
        else:
            plots.append(go.Scatter(x = list(range(2, cycle_num + 1)), y = metrics[i], name = plot_names[i], line_width=2))

    fig = go.Figure(plots)
    
    signal_half_cycle = cut_x_ranges[1][0] - cut_x_ranges[0][0]
    if(add_vertical):
        for i in range(len(cut_x_ranges)-1):
            fig.add_vrect(
                x0=cut_x_ranges[i][0], x1=cut_x_ranges[i + 1][0],
                fillcolor="LightSalmon" if(i %2  == 0) else "LightSkyBlue", 
                opacity=0.2,
                layer="below", line_width=0,
            )

    if(y_range is None):
        y_range = [0, max_val + 5.0 * max_val/100.0]
    fig.update_layout(
        yaxis = dict(
            title = metric_name,
            range = y_range
        ),
        xaxis = dict(
            title = 'Adaptation cycle',
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.7
        ),
        # 'yanchor': 'top'}
    )   

    fig.show()
    fig.write_image("./figures/" + shape_name + ".pdf")
    fig.write_html("./figures/" + shape_name + ".html")


def plot_distributions(x, y, names, colors, x_title, y_title, legend_x_pos, legend_y_pos, file_name = None, is_group = False, show_plot = True, save_plot = True):
    
    data = []
    for i in range(len(x)):
        data.append(go.Box(x = x[i], y =  y[i], name = names[i], marker_color = colors[i], boxmean='sd'))
    
    fig = go.Figure(data = data)
    fig.update_layout(
        yaxis = dict(title = y_title),
        xaxis = dict(title = x_title),
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            yanchor="top",
            y=legend_y_pos,
            xanchor="left",
            x=legend_x_pos
        )
    ) 
    if(is_group):
        fig.update_layout(boxmode='group')       
    
    if(show_plot):
        fig.show()
    
        if(save_plot):
            if(file_name is not None):
                fig.write_image("./figures/" + file_name + ".pdf")
                fig.write_html("./figures/" + file_name + ".htm")