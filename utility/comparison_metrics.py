import os
import cv2
import numpy as np
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import torch
import statistics


def flat_list(list):
  flat_list = [statistics.median(item) for item in list] # 0 = median, 1 = min, 2 = max
  return flat_list

def moving_average(a, n=30) :
  ret = np.cumsum(a, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return ret[n - 1:] / n

def double_lineplot(xs, ys_1, ys_2, title, path='', xaxis='episodes', yaxis='rewards'):
  data = []
  data.append(Scatter(x=xs, y=ys_1, name="PlaNet with regularizer"))
  data.append(Scatter(x=xs, y=ys_2, name="Planet without regularizer"))

  plotly.offline.plot({
    'data': data,
    'layout': dict(title=title, xaxis={'title': xaxis}, yaxis={'title': yaxis})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)

metrics_1 = torch.load('con_reg.pth') # PLANET PURO
metrics_2 = torch.load('senza_reg.pth')

print(metrics_1.keys())
x1 = list(range(0,len(metrics_1['train_rewards'])))



diff_1 = np.absolute(np.array(metrics_1['train_rewards']) - np.array(metrics_1['predicted_rewards']))
rel_diff_1 = moving_average(diff_1 / np.array(metrics_1['train_rewards'])) #diff relativa
diff_assoluta_1 =  moving_average(diff_1)
pure_reward_1 =  flat_list(metrics_1['test_rewards'])

diff_2 = np.absolute(np.array(metrics_2['train_rewards']) - np.array(metrics_2['predicted_rewards']))
rel_diff_2 = moving_average(diff_2 / np.array(metrics_2['train_rewards'])) 
diff_assoluta_2 =  moving_average(diff_2)
pure_reward_2 =  flat_list(metrics_2['test_rewards'])

double_lineplot(x1, pure_reward_1, pure_reward_2, 'plot_pure_reward')
double_lineplot(x1, diff_assoluta_1, diff_assoluta_2, 'plot_con_diff_assoluta')
double_lineplot(x1, rel_diff_1, rel_diff_2, 'plot_con_diff_relativa')




