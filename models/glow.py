import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

class Actnorm(nn.Module):
    def __init__(self, param_dim=(1,2)):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(param_dim))
        self.bias = nn.Parameter(torch.zeros(param_dim))
        self.register_buffer('initialized', torch.tensor(0).byte())

    def forward(self, x, label=None):
        if not self.initialized:

            self.bias.squeeze().data.copy_(x.transpose(0,1).flatten(1).mean(1)).view_as(self.scale)
            self.scale.squeeze().data.copy_(x.transpose(0,1).flatten(1).std(1, unbiased=False) + 1e-6).view_as(self.bias)
            self.initialized += 1

        z = (x - self.bias) / self.scale
        logdet = - self.scale.abs().log().sum() 
        return z, logdet

    def inverse(self, z, label=None):
        x = z * self.scale + self.bias
        logdet = self.scale.abs().log().sum()
        return x, logdet


class Invertible1x1Conv(nn.Module):
    def __init__(self, dim=2):
        super().__init__()

        w = torch.randn(dim, dim)
        w = torch.qr(w)[0]   
        self.w = nn.Parameter(w)

    def forward(self, x, label=None):        
        logdet = torch.slogdet(self.w)[-1]
        return x @ self.w.t(), logdet  

    def inverse(self, z, label=None):
        w_inv = self.w.t().inverse()
        logdet = - torch.slogdet(self.w)[-1]
        return z @ w_inv, logdet

class AffineCoupling(nn.Module):
    def __init__(self, dim=2, width=128, y_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(dim//2 + y_dim, width, bias=True)  
        self.fc2 = nn.Linear(width, width, bias=True)
        self.fc3 = nn.Linear(width, dim, bias=True)
        self.log_scale_factor = nn.Parameter(torch.zeros(1,2))
        
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()

    def forward(self, x, label):
        x_a, x_b = x.chunk(2, dim=1)  
        
        h = F.relu(self.fc1(torch.cat([x_b, label],dim=-1)))
        h = F.relu(self.fc2(h))
        h = self.fc3(h) * self.log_scale_factor.exp()
        t = h[:,0::2] 
        s = h[:,1::2]  
        s = torch.sigmoid(s + 2.)

        z_a = s * x_a + t
        z_b = x_b
        z = torch.cat([z_a, z_b], dim=1) 
        logdet = s.log().sum(1)

        return z, logdet

    def inverse(self, z, label):
        z_a, z_b = z.chunk(2, dim=1)  
        
        h = F.relu(self.fc1(torch.cat([z_b, label], dim=-1)))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)  * self.log_scale_factor.exp()
        t = h[:,0::2] 
        s = h[:,1::2]  
        s = torch.sigmoid(s + 2.)

        x_a = (z_a - t) / s
        x_b = z_b
        x = torch.cat([x_a, x_b], dim=1) 

        logdet = - s.log().sum(1)
        return x, logdet

class FlowSequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, label):
        slogdet = 0.
        for module in self:
            x, logdet = module(x, label)
            slogdet = slogdet + logdet
        return x, slogdet

    def inverse(self, z, label):
        slogdet = 0.
        for module in reversed(self):
            z, logdet = module.inverse(z, label)
            slogdet = slogdet + logdet
        return z, slogdet


class FlowStep(FlowSequential):
    def __init__(self, dim=2, width=128, y_dim=4):
        super().__init__(
                        Actnorm(param_dim=(1,dim)),
                        Invertible1x1Conv(dim=dim),
                        AffineCoupling(dim=dim, width=width, y_dim=y_dim))



class Glow(nn.Module):
    def __init__(self, width=128, depth=10, data_dim=2, y_dim=4):
        super().__init__()

        self.flowstep = FlowSequential(*[FlowStep(dim=data_dim, width=width, y_dim=y_dim) for _ in range(depth)])

        self.register_buffer('base_dist_mean', torch.zeros(2))
        self.register_buffer('base_dist_var', torch.eye(2))

    def forward(self, x, label):
        z, logdet = self.flowstep(x, label)

        return z, logdet

    def inverse(self, z, label):

        x, logdet = self.flowstep.inverse(z, label)

        return x, logdet
    
    def sample(self, z, label):
        with torch.no_grad():
            sample, _ = self.inverse(z, label)
            return sample

    def log_prob(self, x, label):
        z, logdet = self.forward(x, label)
        log_prob = self.base_dist.log_prob(z) + logdet
        return log_prob.unsqueeze(1)


    @property
    def base_dist(self):
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)
    
    