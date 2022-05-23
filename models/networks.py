import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch import nn
import torch.distributions as D
from lib.utils import truncated_normal, reduce_tensor, standard_normal_logprob
from models.chart import ChartPredictor3d, ChartGenerator
from models.AF import AF
from models.flow import get_latent_cnf

class Encoder(nn.Module):
    def __init__(self, s_X_dim, input_dim=3):
        super(Encoder, self).__init__()
        self.s_X_dim = s_X_dim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, s_X_dim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, s_X_dim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
        m = self.fc3_m(m)
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = self.fc3_v(v)

        return m, v


# Model
class ChartPointFlow(nn.Module):
    def __init__(self, args):
        super(ChartPointFlow, self).__init__()
        h_dims_AF = tuple(map(int, args.h_dims_AF.split("-")))

        self.input_dim = 3
        self.s_X_dim = args.s_X_dim
        self.y_dim = args.y_dim
        self.distributed = args.distributed
        self.use_gumbel = args.use_gumbel

        self.mu = args.mu
        self.lmd = args.lmd
        
        self.encoder = Encoder(s_X_dim=args.s_X_dim, input_dim=3)
        self.prior_f = get_latent_cnf(args)
        self.point_g = AF(args.n_flow_AF, args.s_X_dim, h_dims_AF, args.y_dim, args.nonlinearity)
        self.cp = ChartPredictor3d(self.input_dim, args.cp_hdim, args.y_dim, args.tempe, self.use_gumbel)
        self.cg = ChartGenerator(self.s_X_dim, args.y_dim)

        self.KLDIV = nn.KLDivLoss(reduction="batchmean")

    @staticmethod
    def sample_gaussian(size, gpu=None):
        y = torch.randn(*size).float()
        y = y if gpu is None else y.cuda(gpu)
        return y

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.size()).to(mean)
        return mean + std * eps

    @staticmethod
    def gaussian_entropy(logvar):
        const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
        ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
        return ent
 

    def entropy(self, q_y_x):
        H_y_x = -torch.mul(q_y_x, torch.log(q_y_x + 1e-20)).sum(2, keepdim=True).sum(1, keepdim=True).mean()
        return H_y_x

    def Mutual_info(self, q_y_x):
        p_y = q_y_x.sum(1, keepdim=True) / 2048
        H_y = - torch.mul(p_y, torch.log(p_y + 1e-20)).sum(-1) * 2048
        H_y = H_y.mean()

        H_y_x = self.entropy(q_y_x)
        
        return H_y * self.mu - H_y_x * self.lmd
    

    def multi_gpu_wrapper(self, f):
        self.encoder = f(self.encoder)
        self.point_g = f(self.point_g)
        self.prior_f = f(self.prior_f)
        self.cp = f(self.cp)
        self.cg = f(self.cg)

    def make_optimizer(self, args):
        def _get_opt_(params):
            optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2),
                                    weight_decay=args.weight_decay)
            return optimizer
        opt = _get_opt_(list(self.encoder.parameters()) + list(self.point_g.parameters())
                        + list(list(self.prior_f.parameters())) + list(self.cp.parameters()) + list(self.cg.parameters()))
        return opt

    def set_initialized(self, switch):
        self.point_g.module.set_initialized(switch)
        print('is_initialized in actnorm is set to ' + str(switch))

    def forward(self, x, opt, step=None, writer=None, init=False, valid=False):
        opt.zero_grad()
        batch_size = x.size(0)
        num_points = x.size(1)
        s_X_mu, s_X_sigma = self.encoder(x)
        s_X = self.reparameterize_gaussian(s_X_mu, s_X_sigma)

        # Compute H[Q(s_X|X)]
        entropy = self.gaussian_entropy(s_X_sigma)

        # Compute the prior probability p(w)
        w, delta_log_pw = self.prior_f(s_X, None, torch.zeros(batch_size, 1).to(s_X))
        log_pw = standard_normal_logprob(w).view(batch_size, -1).sum(1, keepdim=True)
        delta_log_pw = delta_log_pw.view(batch_size, 1)
        log_ps_X = log_pw - delta_log_pw

        
        s_X_new = s_X.view(*s_X.size())
        s_X_new = s_X_new + (log_ps_X * 0.).mean()

        # Sampling labels
        input_feature = s_X.clone().detach()
        label, probs = self.cp(x, input_feature)
        
        # Compute the reconstruction likelihood p(X|y,s_X)
        z, delta_log_pz = self.point_g(x, s_X_new, label)
        log_pz = standard_normal_logprob(z).view(batch_size, -1).sum(1, keepdim=True)
        delta_log_pz = delta_log_pz.view(batch_size, num_points, 1).sum(1)
        log_px = log_pz - delta_log_pz

        s_X_in = s_X.clone().detach()
        posterior = self.cg(s_X_in) + 1e-20
        rate = probs.sum(1).view(batch_size, -1).clone().detach()
        rate = rate / 2048 + 1e-20

        # Loss
        entropy_loss = -entropy.mean()
        recon_loss = -log_px.mean()
        prior_loss = -log_ps_X.mean()
        entropy_ = -self.entropy(probs)
        mutual_loss = -self.Mutual_info(probs)
        posterior_loss = self.KLDIV(rate.log(), posterior)
        loss = entropy_loss + prior_loss + recon_loss  + posterior_loss + mutual_loss + entropy_
        if not init and not valid:
            loss.backward()
            opt.step()
        # LOGGING (after the training)
        if self.distributed:
            loss = reduce_tensor(loss.mean())
            entropy_log = reduce_tensor(entropy.mean())
            recon = reduce_tensor(-log_px.mean())
            prior = reduce_tensor(-log_ps_X.mean())
            mutual = reduce_tensor(mutual_loss)
            pos = reduce_tensor(posterior_loss)
        else:
            loss = loss.mean()
            entropy_log = entropy.mean()
            recon = -log_px.mean()
            prior = -log_ps_X.mean()
            mutual = mutual_loss
            pos = posterior_loss

        recon_nats = recon / float(x.size(1) * x.size(2))
        prior_nats = prior / float(self.s_X_dim)
        mutual_nats = mutual / float(x.size(1))

        if writer is not None and not valid:
            writer.add_scalar('train/entropy', entropy_log, step)
            writer.add_scalar('train/prior', prior, step)
            writer.add_scalar('train/prior(nats)', prior_nats, step)
            writer.add_scalar('train/recon', recon, step)
            writer.add_scalar('train/recon(nats)', recon_nats, step)
            writer.add_scalar('train/loss', loss.item(), step)
            writer.add_scalar('train/mutual', mutual, step)
            writer.add_scalar('train/posterior', pos, step)
            

        return {
            'entropy': entropy_log.cpu().detach().item(),
            'prior_nats': prior_nats.cpu().detach().item(),
            'recon_nats': recon_nats.cpu().detach().item(),
            'prior': prior.cpu().detach().item(),
            'recon': recon.cpu().detach().item(),
            'loss': loss.item(),
            'mutual': mutual_nats.cpu().detach().item(),
            'pos': pos.cpu().detach().item(),
        }

    def encode(self, x):
        s_X_mu, s_X_sigma = self.encoder(x)
        s_X =  self.reparameterize_gaussian(s_X_mu, s_X_sigma)
        label, _ = self.cp(x, s_X)
        return s_X, label

    def decode(self, s_X, label, num_points):
        # transform points from the prior to a point cloud, conditioned on a shape code and labels
        z = self.sample_gaussian((s_X.size(0), num_points, self.input_dim))
        x = self.point_g(z, s_X, label, reverse=True).view(*z.size())
        return z, x

    def sample(self, batch_size, num_points, gpu=None):
        # Generate the shape code from the prior
        w = self.sample_gaussian((batch_size, self.s_X_dim), gpu=gpu)
        s_X = self.prior_f(w, None, reverse=True)
        # Sample labels from the shape code
        posterior = self.cg(s_X).unsqueeze(1).repeat(1, num_points, 1)
        p_y = D.OneHotCategorical(probs=posterior)
        label = p_y.sample()
        # Sample points conditioned on the shape code and labels
        z = self.sample_gaussian((batch_size, num_points, self.input_dim), gpu=gpu) 
        x = self.point_g(z, s_X, label, reverse=True).view(*z.size())
        return s_X, x, label

    def reconstruct(self, x, num_points=None):
        num_points = x.size(1) if num_points is None else num_points
        s_X, label = self.encode(x)
        _, x = self.decode(s_X, label, num_points)
        return x, label

    def super_resolution(self, x, num_points=None):
        z, _ = self.encode(x)
        posterior = self.chart_posterior(z).unsqueeze(1).repeat(1, num_points, 1)
        p_y = D.OneHotCategorical(probs=posterior)
        label = p_y.sample()
        _, x = self.decode(z, label, num_points)
        return x, label



             
        


