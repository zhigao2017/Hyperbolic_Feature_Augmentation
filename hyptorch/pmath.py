"""
Implementation of various mathematical operations in the Poincare ball model of hyperbolic space. Some
functions are based on the implementation in https://github.com/geoopt/geoopt (copyright by Maxim Kochurov).
"""

import numpy as np
import torch
from scipy.special import gamma


EPS = {torch.float32: 1e-8, torch.float64: 1e-8}

def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


# +
class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-6, 1 - 1e-6)
        ctx.save_for_backward(x)
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)
    
    
class RiemannianGradient(torch.autograd.Function):
    
    c = 1
    
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        #x: B x d

        scale = (1 - RiemannianGradient.c * x.pow(2).sum(-1, keepdim=True)).pow(2) / 4
        return grad_output * scale



# -

class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x + torch.sqrt_(1 + x.pow(2))).clamp_min_(1e-6).log_()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


def arctanh(x):
    return Artanh.apply(x)


def arcsinh(x):
    return Arsinh.apply(x)


def arccosh(x, eps=1e-5):  # pragma: no cover
    x = x.clamp(-1 + eps, 1 - eps)
    return torch.log(x + torch.sqrt(1 + x) * torch.sqrt(x - 1))



def sink (x,k):
    if k >= 0:
        return torch.sin(x)
    else:
        return torch.sinh(x)

def cosk (x,k):
    if k >= 0:
        return torch.cos(x)
    else:
        return torch.cosh(x)

def tank (x,k):
    if k >= 0:
        return torch.tan(x)
    else:
        return torch.tanh(x)


def arcsink (x,k):
    if k >= 0:
        return torch.asin(x)
    else:
        return arcsinh(x)

def arccosk (x,k):
    if k >= 0:
        return torch.acos(x)
    else:
        return arccosh(x)

def arctank (x,k):
    if k >= 0:
        return torch.atan(x)
    else:
        return arctanh(x)




def mobius_add(x, y, *, k=1.0):

    k = torch.as_tensor(k).type_as(x)
    return _mobius_add(x, y, k)


def _mobius_add(x, y, k):
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    y2 = y.pow(2).sum(dim=-1, keepdim=True)
    #print('x2',x2)
    #print('y2',y2)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 - 2 * k * xy - k * y2) * x + (1 + k * x2) * y
    denom = 1 - 2 * k * xy + k ** 2 * x2 * y2
    return num / (denom + 1e-5)




def expmap(x, u, *, k=1.0):

    k = torch.as_tensor(k).type_as(x)
    return _expmap(x, u, k)


def _expmap(x, u, k):  # pragma: no cover
    sqrt_k = torch.abs(k) ** 0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-6)
    second_term = (
            tank(sqrt_k / 2 * lambda_x(x, k, keepdim=True) * u_norm, k)
            * u
            / (sqrt_k * u_norm)
    )
    #print('expmap lambda_x',lambda_x(x, k, keepdim=True))
    #print('second_term',second_term)
    #gamma_1 = _mobius_add(x, second_term, k)
    gamma_1 = _mobius_addition_batch(x, second_term, k)
    return gamma_1


def expmap0(u, *, k=1.0):

    k = torch.as_tensor(k).type_as(u)
    return _expmap0(u, k)


def _expmap0(u, k):
    sqrt_k = torch.abs(k) ** 0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-6)
    gamma_1 = tank(sqrt_k * u_norm,k) * u / (sqrt_k * u_norm)
    #print('tank(sqrt_k * u_norm,k)',tank(sqrt_k * u_norm,k))
    return gamma_1


def logmap(x, y, *, k=1.0):

    k = torch.as_tensor(k).type_as(x)
    return _logmap(x, y, k)


def _logmap(x, y, k):  # pragma: no cover
    #sub = _mobius_add(-x, y, k) 
    #print('-------------')
    #print('x',x.shape)
    #print('y',y.shape)
    sub = _mobius_addition_batch(-x, y, k)
    #print('sub',sub.shape)
    sub_norm = sub.norm(dim=-1, p=2, keepdim=True)
    lam = lambda_x(x, k, keepdim=True)
    sqrt_k = torch.abs(k) ** 0.5
    return 2 / sqrt_k / lam * arctank(sqrt_k * sub_norm, k) * sub / sub_norm


def logmap0(y, *, k=1.0):

    k = torch.as_tensor(k).type_as(y)
    return _logmap0(y, k)


def _logmap0(y, k):
    sqrt_k = torch.abs(k) ** 0.5
    y_norm = torch.clamp_min(y.norm(dim=-1, p=2, keepdim=True), 1e-6)
    return y / y_norm / sqrt_k * arctank(sqrt_k * y_norm, k)


def _tensor_dot(x, y):
    res = torch.einsum('ij,kj->ik', (x, y))
    return res


def _mobius_addition_batch(x, y, k):
    xy = _tensor_dot(x, y)  # B x C
    x2 = x.pow(2).sum(-1, keepdim=True)  # B x 1
    y2 = y.pow(2).sum(-1, keepdim=True)  # C x 1
    num = (1 - 2 * k * xy - k * y2.permute(1, 0))  # B x C
    num = num.unsqueeze(2) * x.unsqueeze(1)
    num = num + (1 + k * x2).unsqueeze(2) * y  # B x C x D
    denom_part1 = 1 - 2 * k * xy  # B x C
    denom_part2 = k ** 2 * x2 * y2.permute(1, 0)
    denom = denom_part1 + denom_part2
    res = num / (denom.unsqueeze(2) + 1e-5)
    return res



def p2k(x, k):
    denom = 1 - k * x.pow(2).sum(-1, keepdim=True)
    return 2 * x / denom


def k2p(x, k):
    denom = 1 + torch.sqrt(1 + k * x.pow(2).sum(-1, keepdim=True))
    return x / denom


def lorenz_factor(x, *, k=1.0, dim=-1, keepdim=False):

    return 1 / torch.sqrt( (1 + k * x.pow(2).sum(dim=dim, keepdim=keepdim)).clamp_min_(1e-6))


def poincare_mean(x, dim=0, k=1.0):
    x = p2k(x, k)
    lamb = lorenz_factor(x, k=k, keepdim=True)
    mean = torch.sum(lamb * x, dim=dim, keepdim=True) / (torch.sum(lamb, dim=dim, keepdim=True)).clamp_min_(1e-6)
    mean = k2p(mean, k)
    return mean.squeeze(dim)


def tangent_mean(x, dim=0, k=1.0):
    x = logmap0(x,k=k)
    mean = torch.mean(x, dim=0, keepdim=True)
    mean = expmap0(mean,k=k)
    return mean.squeeze(dim)

def _dist_matrix(x, y, k):
    sqrt_k = torch.abs(k) ** 0.5
    return 2 / sqrt_k * arctank(sqrt_k * torch.norm(_mobius_addition_batch(-x, y, k=k), dim=-1), k)


def dist_matrix(x, y, k=1.0):
    k = torch.as_tensor(k).type_as(x)
    return _dist_matrix(x, y, k)




def frechet_variance( x, mu, k, w=None):
    """
    Args
    ----
        x (tensor): points of shape [..., points, dim]
        mu (tensor): mean of shape [..., dim]
        w (tensor): weights of shape [..., points]

        where the ... of the three variables line up
    
    Returns
    -------
        tensor of shape [...]
    """
    distance = dist_matrix(x, mu.unsqueeze(-2), k=k).squeeze()
    if w is None:
        return distance.mean(dim=-1)
    else:
        return (distance * w).sum(dim=-1)



def _gyration(u, v, w, k):
    u2 = u.pow(2).sum(dim=-1, keepdim=True)
    v2 = v.pow(2).sum(dim=-1, keepdim=True)
    uv = (u * v).sum(dim=-1, keepdim=True)
    uw = (u * w).sum(dim=-1, keepdim=True)
    vw = (v * w).sum(dim=-1, keepdim=True)
    a = - k.pow(2) * uw * v2 - k * vw + 2 * k.pow(2) * uv * vw
    b = - k.pow(2) * vw * u2 + k * uw
    d = 1 - 2 * k * uv + k.pow(2) * u2 * v2
    return w + 2 * (a * u + b * v) / d.clamp_min(EPS[u.dtype])



def lambda_x( x, k, keepdim=False):
    #return 2 / (1 + k * x.pow(2).sum(dim=-1, keepdim=keepdim)).clamp_min(min=EPS[x.dtype])
    #return 2 / (1 + k * x.pow(2).sum(dim=-1, keepdim=keepdim))

    denominator=(1 + k * (x.norm(dim=-1, p=2, keepdim=keepdim)).pow(2))
    repalce=(torch.ones(denominator.shape).cuda())*EPS[x.dtype]
    denominator=torch.where(denominator==0,repalce,denominator)
    
    return 2 / denominator

def transp(x, y, u, k):
    return (
        _gyration(y, -x, u, k)
        * lambda_x(x, k, keepdim=True)
        / lambda_x(y, k, keepdim=True)
    )


def RiemannianBatchNorm(x, updates,running_mean, running_var, bn_weight, bn_biased, k, training=True, momentum=0.9):

    #on_manifold = self.man.exp0(bn_weight,k)
    on_manifold = expmap0(bn_weight,k=k)
    #print('in BN after on_manifold',torch.sum(torch.isnan(on_manifold))) 
    if training:
        # frechet mean, use iterative and don't batch (only need to compute one mean)
        #input_mean = frechet_mean(x, self.man, k)
        #print('in BN before x',torch.sum(torch.isnan(x)),x) 
        input_mean = tangent_mean(x,dim=0,k=k)
        #print('in BN after input_mean',torch.sum(torch.isnan(input_mean)),input_mean) 
        input_var = frechet_variance(x, input_mean, k=k).clamp_min_(1e-6)
        #print('in BN after input_var',torch.sum(torch.isnan(input_var)),input_var)  
        # transport input from current mean to learned mean
        #print('in BN after input_logm1',torch.sum(torch.isnan(input_logm)),input_logm)
        # re-scaling
        #print('in BN after input_logm1 bn_biased',bn_biased)
        #print('in BN after input_logm2',',nan,', torch.sum(torch.isnan(input_logm)),',inf,'  ,torch.sum(torch.isinf(input_logm)),input_logm)
        # project back
        trans=(bn_biased / (input_var)).sqrt() * transp( input_mean, on_manifold, logmap(input_mean, x, k=k), k=k)
        output = expmap(on_manifold, trans, k=k)
        #print('in BN after output',torch.sum(torch.isnan(output)),output)
        updates += 1

        running_mean = expmap(
            running_mean,
            (1 - momentum) * logmap(running_mean, input_mean, k=k), k=k
        )
        
        running_var = (
            1 - 1 / updates
        ) * running_var + input_var / updates

    else:
        if updates == 0:
            raise ValueError("must run training at least once")

        #input_mean = frechet_mean(x, self.man, k)
        #print('in BN, x',',nan,', torch.sum(torch.isnan(x)),',inf,',torch.sum(torch.isinf(x)),x)
        input_mean = tangent_mean(x,dim=0,k=k)
        input_var = frechet_variance(x, input_mean, k=k).clamp_min_(1e-6)


        #print('in BN, input_mean',',nan,', torch.sum(torch.isnan(input_mean)),',inf,',torch.sum(torch.isinf(input_mean)),input_mean)
        #print('in BN, running_mean',',nan,', torch.sum(torch.isnan(running_mean)),',inf,',torch.sum(torch.isinf(running_mean)),running_mean)
        #print('in BN, logmap(input_mean, x, k=k)',',nan,', torch.sum(torch.isnan(logmap(input_mean, x, k=k))),',inf,',torch.sum(torch.isinf(logmap(input_mean, x, k=k))),logmap(input_mean, x, k=k))

        input_logm = transp(
            input_mean,
            running_mean,
            logmap(input_mean, x, k=k), k=k
        )
        #print('in BN after input_logm',',nan,', torch.sum(torch.isnan(input_logm)),',inf,',torch.sum(torch.isinf(input_logm)),input_logm)

        assert not torch.any(torch.isnan(input_logm))

        # re-scaling
        input_logm = (
            running_var / (x.shape[0] / (x.shape[0] - 1) * input_var )
        ).sqrt() * input_logm

        # project back
        output = expmap(on_manifold, input_logm, k=k)

    return output