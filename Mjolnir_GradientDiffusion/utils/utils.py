import numpy as np
import torch


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)
import torch
import torch.nn.functional as F


def label_to_onehot(target, num_classes=100):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


def label_to_onehot2(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def weight_init(m):
    try:
        if hasattr(m, 'weight'):
            m.weight.data.uniform_(-0.5, 0.5)


    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())

    try:
        if hasattr(m, 'bias'):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())


def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    """Sample at any time t to calculate the loss"""
    batch_size = x_0.shape[0]

    # Generate random moments t for a batch size of samples
    t = torch.randint(0, n_steps, size=(batch_size // 2,)).to(device)
    t = torch.cat([t, n_steps - 1 - t], dim=0)[:batch_size]

    # x0
    a = alphas_bar_sqrt[t].view(-1, 1)

    # eps
    aml = one_minus_alphas_bar_sqrt[t].view(-1, 1)

    # randon noise eps
    e = torch.randn_like(x_0).to(device)

    # model input
    x = x_0 * a + e * aml

    # Feed into the model to obtain the random noise prediction at moment t.
    output = model(x, t)


    return (e - output).square().mean()


# （n，2）
def channel_deal_2(channel: list):
    if (len(channel) % 2 == 0):
        MN = int(len(channel) / 2)
    else:
        MN = (int(len(channel)) + 1 / 2)
        channel.extend([np.mean(channel)])
    channel = np.array(channel)
    maxChannel = channel.reshape([MN, 2])
    return maxChannel


# (10000, 2)
def channel_deal_3(channel: list):
    pad = int(14400 - len(channel))
    channel.extend([np.mean(channel) for _ in range(pad)])
    channel = np.array(channel)
    maxchannel = channel.reshape([120, 120])
    return maxchannel

def channel_deal_4(channel):
    # new_grad = torch.randn(16384).to(device)
    grad = torch.cat([g.flatten() for g in channel])
    if len(grad) > 16384:
        new_grad = torch.randn(92416).to(device)
    else:
        new_grad = torch.randn(16384).to(device)
    mean_value = grad.mean()
    new_grad[:len(grad)] = grad
    new_grad[len(grad):] = mean_value
    w = int(np.sqrt(len(new_grad)))
    new_grad = new_grad.view(w, w)
    # new_grad = new_grad.view(128, 128)
    return new_grad
# DP
def cal_sensitivity(lr, clip, dataset_size):
    return 2 * lr * clip / dataset_size


def Gaussian_Simple(epsilon, delta):
    return np.sqrt(2 * np.log(1.25 / delta)) / epsilon


def per_sample_clip(net, clipping, norm=2):
    grad_samples = [x.grad_sample for x in net.parameters()]
    per_param_norms = [
        g.reshape(len(g), -1).norm(norm, dim=-1) for g in grad_samples
    ]
    per_sample_norms = torch.stack(per_param_norms, dim=1).norm(norm, dim=1)
    per_sample_clip_factor = (
        torch.div(clipping, (per_sample_norms + 1e-6))
    ).clamp(max=1.0)
    for grad in grad_samples:
        factor = per_sample_clip_factor.reshape(per_sample_clip_factor.shape + (1,) * (grad.dim() - 1))
        grad.detach().mul_(factor.to(grad.device))
    # average per sample gradient after clipping and set back gradient
    for param in net.parameters():
        param.grad = param.grad_sample.detach().mean(dim=0)


def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
    """from x[T] recover x[T-1]、x[T-2]|...x[0]"""
    cur_x = torch.randn(shape).to(device)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq


def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    """Sample the reconstruction value at time t from x[T]"""
    t = torch.tensor([t]).to(device)
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x, t)
    mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))
    z = torch.randn_like(x).to(device)
    sigma_t = betas[t].sqrt()
    sample = mean + sigma_t * z
    return (sample)
