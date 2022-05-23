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
parser.add_argument('--iter', default=36000, type=int, help='number of training iterations')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--data', default='circle', type=str, help='2D dataset to use')
parser.add_argument('--vis_freq', default=500, type=int, help='visualize frequency') 
parser.add_argument('--val_freq', type=int, default=400)

parser.add_argument('--width', default=128, type=int, help='width of the glow model') 
parser.add_argument('--depth', default=20, type=int, help='depth of the glow model') 
parser.add_argument('--x_dim', default=2, type=int, help='data dimension') 
parser.add_argument('--y_dim', default=4, type=int, help='number of charts') 
parser.add_argument('--h_dim', default=256, type=int, help='ChartPredictor hidden dim')

parser.add_argument('--mu', default=1.1, type=float, help='Mutual information coefficient')
parser.add_argument('--lmd', default=1.1, type=float, help='Mutual information coefficient')

parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
args = parser.parse_args()

save_path = 'results/' + args.data
save_path_img = os.path.join(save_path, "img")
if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.makedirs(save_path_img)


if __name__ == '__main__':
    model = Glow(width=args.width, depth=args.depth, data_dim=args.x_dim, y_dim=args.y_dim).cuda()
    cp = ChartPredictor2d(args.x_dim, args.h_dim, args.y_dim).cuda()
    
    optimizer = optim.Adam(list(model.parameters()) + list(cp.parameters()), lr=args.lr, betas=(args.b1, args.b2))
    
    total_loss = 0
    best_loss = float('inf')
    batch_size = args.batch_size
    for itr in range(1, args.iter + 1):
        optimizer.zero_grad()
        x = get_toydata(args.data, batch_size)
        x = torch.from_numpy(x).type(torch.float32).cuda()
        label, probs = cp(x)
        loss = -model.log_prob(x, label).sum() - entropy(probs) - Mutual_Information(probs, args.mu, args.lmd, batch_size)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        print("\ritr : {} | loss : {}".format(itr, loss.item()), end="")
        
        if itr % args.vis_freq == 0:
            with torch.no_grad():
                model.eval()
                new_sampled_z = torch.randn(2000,2).cuda()
                y = D.OneHotCategorical(probs=( (1 / args.y_dim) * torch.ones(2000, args.y_dim))).sample().cuda()
                predicted_x = model.sample(new_sampled_z, y)
                plt.figure(figsize = (5,5))
                plt.xlim(-5, 5)
                plt.ylim(-5, 5)
                plt.scatter(predicted_x[:,0].cpu().detach(), predicted_x[:,1].cpu().detach(), s=5)
                plt.axes().set_aspect('equal', 'datalim')
                fig_filename = os.path.join(save_path_img, '{:04d}.jpg'.format(itr))
                plt.savefig(fig_filename, format='jpg', dpi=300)
                plt.close()

                fig_filename = os.path.join(save_path_img, '{:04d}_chart.jpg'.format(itr))
                visualize_chart_2d(predicted_x, y, fig_filename)

                model.train()
        
        if itr % args.val_freq == 0:
            with torch.no_grad():
                model.eval()
                
                x = get_toydata(args.data, 1000)
                x = torch.from_numpy(x).type(torch.float32).cuda()
                label, probs = cp(x)
                test_loss = -model.log_prob(x, label).sum()- entropy(probs) - Mutual_Information(probs, args.mu, args.lmd, 1000)
                print("\nvalid loss : {} | best loss : {}".format(test_loss, best_loss))
                if test_loss.item() < best_loss:
                    best_loss = test_loss.item()
                    path=os.path.join(save_path, "ckp-best.pt")
                    torch.save({"model" : model.state_dict(),
                                "cp" : cp.state_dict(),}, path)
                    print("best model saved | itr : {}".format(itr))
                model.train()
