import torch
import numpy as np
import random
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import log, pi
import torch.distributed as dist

class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def Mutual_Information(q_y_x, mu, lmd, batch_size):
    q_y = q_y_x.sum(-2) / batch_size
    H_y = - torch.mul(q_y, torch.log(q_y + 1e-20))
    H_y = H_y.sum(-1) * batch_size
    
    H_y_x = entropy(q_y_x)

    return H_y * mu - H_y_x * lmd

def entropy(q_y_x):
    H_y_x = -torch.mul(q_y_x, torch.log(q_y_x + 1e-20)).sum(-1).sum(-1)
    return H_y_x

def set_random_seed(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def visualize_point_clouds(pts, gtr, idx, pert_order=[0, 1, 2]):
    R = 180/255
    G = 30/255
    B = 45/255
    alpha = 0.4
    COLOR = [[R, G, B, alpha]]

    pts = pts.cpu().detach().numpy()[:, pert_order]
    gtr = gtr.cpu().detach().numpy()[:, pert_order]

    fig = plt.figure(figsize=(6, 3))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Sample:%s" % idx)
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=COLOR, s=0.1)

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Ground Truth:%s" % idx)
    ax2.scatter(gtr[:, 0], gtr[:, 1], gtr[:, 2], c=COLOR, s=0.1)

    ax1.set_xlim(ax2.get_xlim())
    ax1.set_ylim(ax2.get_ylim())
    ax1.set_zlim(ax2.get_zlim())

    ax1.axis('off')
    ax2.axis('off')
    fig.canvas.draw()

    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.close()
    
    return res

def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size is None:
        world_size = dist.get_world_size()

    rt /= world_size
    return rt

def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * log(2 * pi)
    return log_z - z.pow(2) / 2

# Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
def truncated_normal(tensor, mean=0, std=1, trunc_std=2):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def save(model, optimizer, epoch, scheduler, valid_loss, log_dir, tot_duration, path):
    d = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'valid_loss': valid_loss,
        'log_dir': log_dir,
        'tot_duration': tot_duration,
    }
    torch.save(d, path)


def resume(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location='cpu')
    try:
        model.load_state_dict(ckpt['model'], strict=True)
    except RuntimeError:
        print("INFO: this model is trained with DataParallel. Creating new state_dict without module...")
        state_dict = ckpt["model"]
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['epoch']
    valid_loss = ckpt['valid_loss']
    log_dir = ckpt['log_dir']
    tot_duration = ckpt['tot_duration']

    return model, optimizer, scheduler, start_epoch, valid_loss, log_dir, tot_duration

def visualize_chart(x, y, fig_filename):
    x = x[:, [0, 2, 1]]
    x = x.unsqueeze(0)
    color = ['red', 'blue', 'green', 'orange', 'c', 'm', 'k', 'purple','gray', 'tan', 'maroon', 'pink', "indianred", "tomato", "springgreen", "darkslateblue", "midnightblue", "olive", "dodgerblue", "rosybrown", "saddlebrown", "crimson", "thistle", "yellow", "burlywood", "darkred", "sandybrown", "slateblue"]
    label = y.clone().unsqueeze(0)
    p = sort_func(x, label, 0)
    fig = plt.figure(figsize = (10, 10))
    ax = Axes3D(fig)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')

    for k in range(len(p)):
        if y.size(-1) > 28:
            ax.scatter3D(p[k][:,0].cpu().detach().numpy(), p[k][:,1].cpu().detach().numpy(), p[k][:,2].cpu().detach().numpy(), marker="o", s=3)
        else:
            ax.scatter3D(p[k][:,0].cpu().detach().numpy(), p[k][:,1].cpu().detach().numpy(), p[k][:,2].cpu().detach().numpy(), marker="o", color = color[k], s=3)

    plt.savefig(fig_filename)
    plt.close()

def sort_func(point, label, num):
    points = []
    p = []
    for k in range(label.size(2)):
        chart = []
        points.append(chart)
    for k in range(point.size(1)):
        c = int(label[num, k].max(0)[1].cpu().numpy())
        points[c].append(point[num, k])
    for k in range(len(points)):
        if len(points[k]) > 0:
            p.append(torch.cat(points[k]).reshape(len(points[k]), 3))
    return p

def sort_func_2d(point, label, num):
    points = []
    p = []
    for k in range(label.size(2)):
        chart = []
        points.append(chart)
    for k in range(point.size(1)):
        c = int(label[num, k].max(0)[1].cpu().numpy())
        points[c].append(point[num, k])
    for k in range(len(points)):
        if len(points[k]) > 0:
            p.append(torch.cat(points[k]).reshape(len(points[k]), 2))
    return p

def visualize_chart_2d(x, y, save_path, scale=5):
    x = x.unsqueeze(0)
    label = y.clone().unsqueeze(0)
    p = sort_func_2d(x, label, 0)
    fig = plt.figure(figsize = (5,5))
    ax = fig.add_subplot(111)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    
    for k in range(len(p)):
        ax.scatter(p[k][:,0].cpu().detach().numpy(), p[k][:,1].cpu().detach().numpy(), s=5, marker="o")
    plt.xlim(-scale, scale)
    plt.ylim(-scale, scale)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.savefig(save_path, format='jpg', dpi=300)
    plt.close()