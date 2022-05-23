import argparse

import torch
import torch.optim as optim
import torch.distributions as D
import matplotlib.pyplot as plt

from models.glow import Glow
from models.chart import ChartPredictor2d
from lib.utils import Mutual_Information
from lib.utils import entropy
from lib.utils import visualize_chart_2d 
from lib.toydata import get_toydata

import os 

parser = argparse.ArgumentParser(description='Training of Glow (2D)')
parser.add_argument('--data', default='circle', type=str, help='2D dataset to use')

parser.add_argument('--width', default=128, type=int, help='width of the glow model') 
parser.add_argument('--depth', default=20, type=int, help='depth of the glow model') 
parser.add_argument('--x_dim', default=2, type=int, help='data dimension') 
parser.add_argument('--y_dim', default=4, type=int, help='number of charts') 
parser.add_argument('--h_dim', default=256, type=int, help='ChartPredictor hidden dim')

parser.add_argument('--pre_trained', default="./pretrained/toy_data/circle_ckp-best.pt", type=str, help='pretrained model dir')

args = parser.parse_args()

save_path = os.path.join('test', args.data)
if not os.path.exists(save_path):
    os.makedirs(save_path)

if __name__ == '__main__':
    model = Glow(width=args.width, depth=args.depth, data_dim=args.x_dim, y_dim=args.y_dim).cuda()
    cp = ChartPredictor2d(args.x_dim, args.h_dim, args.y_dim).cuda()
    
    ckp = torch.load(args.pre_trained)
    model.load_state_dict(ckp["model"])
    cp.load_state_dict(ckp["cp"])

    with torch.no_grad():
        new_sampled_z = torch.randn(2000,2).cuda()
        y = D.OneHotCategorical(probs=( (1 / args.y_dim) * torch.ones(2000, args.y_dim))).sample().cuda()
        predicted_x = model.sample(new_sampled_z, y)

        plt.figure(figsize = (5,5))
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.scatter(predicted_x[:,0].cpu().detach(), predicted_x[:,1].cpu().detach(), s=5)
        plt.axes().set_aspect('equal', 'datalim')
        fig_filename = os.path.join(save_path, 'generated_sample.jpg')
        plt.savefig(fig_filename, format='jpg', dpi=300)
        plt.close()

        fig_filename = os.path.join(save_path, 'generated_sample_chart.jpg')
        visualize_chart_2d(predicted_x, y, fig_filename)

    