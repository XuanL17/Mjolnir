import torch
from torch import nn
import numpy as np
import math
from inspect import isfunction
from einops.layers.torch import Rearrange
from torchvision.transforms import Compose, Lambda, ToPILImage
import torch.nn.functional as F
from tqdm.auto import tqdm

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(nn.Module):
    def __init__(self, fn):

        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):

        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    """
    This upsampling module doubles the width and height dimensions of the input tensor.
    :param dim:
    :param dim_out:
    :return:
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),            # use nearest-neighbor filling to double the data in length and width
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),    # Then, use convolution to extract local correlations from the doubled data for padding
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    """
    The role of the downsampling module is to reduce the resolution of the input tensor, commonly used to downsample feature maps in deep learning models.
    In this implementation, the downsampling is achieved by using a $2 \times 2$ max pooling operation,
    which halves the width and height of the input tensor, and then the output tensor is obtained using the aforementioned transformations and convolution operations.
    Since this implementation uses shape transformation operations, it avoids the loss of information during downsampling by not using traditional convolution or pooling operations.
    :param dim:
    :param dim_out:
    :return:
    """
    return nn.Sequential(
        # Transform the shape of the input tensor from (batch_size, channel, height, width) to (batch_size, channel * 4, height / 2, width / 2)
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        # Perform a $1 \times 1$ convolution operation on the transformed tensor to reduce the number of channels from dim * 4 (i.e., the number of channels after transformation)
        # to dim (i.e., the specified number of output channels), to obtain the output tensor.
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


def extract(a, t, x_shape):
    """
    Retrieve specific elements from the given tensor a. t is a tensor containing the indices of elements to be retrieved,
    which correspond to elements in the tensor a. The output of this function is a tensor,
    containing the elements in tensor a that correspond to each index in tensor t.
    :param a:
    :param t:
    :param x_shape:
    :return:
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# show a random one
reverse_transform = Compose([
     Lambda(lambda t: (t + 1) / 2),
     Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
     Lambda(lambda t: t * 255.),
     Lambda(lambda t: t.numpy().astype(np.uint8)),
     ToPILImage(),
])

