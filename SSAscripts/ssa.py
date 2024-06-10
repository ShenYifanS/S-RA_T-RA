import numpy as np
import torch
from torch import nn
from torch.autograd import Variable as V
import torch.nn.functional as F
from Normalize import Normalize
# import pretrainedmodels
from tqdm import tqdm


def dct1(x):
    """
    Discrete Cosine Transform, Type I

    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])

    return torch.fft.fft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), 1).real.view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I

    Our definition if idct1 is such that idct1(dct1(x)) == x

    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.fft.fft(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    # V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i
    V = Vc.real * W_r - Vc.imag * W_i
    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    tmp = torch.complex(real=V[:, :, 0], imag=V[:, :, 1])
    v = torch.fft.ifft(tmp)

    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape).real


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_3d(dct_3d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

# def image_transfer(images, device, sigma=4, rho=0.1):
#     # images: size = H, W, 3 range = [0,1]
#     # image_width
#     x = images.clone()
#     gauss = torch.randn(images.size()[0], images.size()[1], 3) * (sigma / 255)
#     gauss = gauss.to(device=device)
#     x_dct = dct_2d(x + gauss).to(device=device)
#     mask = (torch.rand_like(x) * 2 * rho + 1 - rho).to(device=device)
#     x_idct = idct_2d(x_dct * mask)
#     x_idct = V(x_idct, requires_grad=True)
#
#     return x_idct




# def image_transfer(images, box, device, sigma=4, rho=0.1):
# def image_transfer(images, device, sigma=6, rho=0.15):
#     """
#     Apply image transformations on a specific region defined by box.
#
#     :param images: Tensor representing the image, size = H, W, C (height, width, channels)
#     :param box: Tuple defining the region for transformation (minx, maxx, miny, maxy)
#     :param device: The device (CPU or GPU) to perform computations
#     :param sigma: Standard deviation for Gaussian noise
#     :param rho: Range parameter for mask
#     :return: Image tensor with transformations applied on the specified region
#     """
#     x = images.clone()
#     # TODO: should clip here
#     # Extract the coordinates from the box
#     # clip and normalize here
#     region = x/255
#
#     # Generate Gaussian noise with the correct shape for the region
#     gauss = torch.randn(region.size()[0], region.size()[1], region.size()[2]) * (sigma / 255)
#     gauss = gauss.to(device=device)
#
#     # Apply the transformations on the extracted region
#     region_dct = dct_2d(region + gauss).to(device=device)
#     mask = (torch.rand_like(region) * 2 * rho + 1 - rho).to(device=device)
#     region_idct = idct_2d(region_dct * mask)
#     region_idct = torch.autograd.Variable(region_idct, requires_grad=True)
#
#
#     print("region_idctMax: ", region_idct.max(), " ;region_idctMin: ", region_idct.min())
#     # region_idct = torch.clamp(region_idct, 0, 255)
#     # region_idct = (region_idct - region_idct.min()) / (region_idct.max() - region_idct.min()) * 255
#
#     # Place the transformed region back into the image
#     # x[scaled_miny:scaled_maxy, scaled_minx:scaled_maxx, :] = region_idct
#     x = region *255
#     print("Max: ", x.max(), " ;Min: ", x.min())
#     # TODO: scale back
#     return x



def image_transfer(predictor, input_points, input_label, images, ε, device, sigma=16, rho=0.1):
    """
    Apply image transformations on a specific region defined by box.

    :param images: Tensor representing the image, size = H, W, C (height, width, channels)
    :param box: Tuple defining the region for transformation (minx, maxx, miny, maxy)
    :param device: The device (CPU or GPU) to perform computations
    :param sigma: Standard deviation for Gaussian noise
    :param rho: Range parameter for mask
    :return: Image tensor with transformations applied on the specified region
    """
    x = images.clone()
    # x = x.permute(2, 0, 1)
    # TODO: should clip here
    # Extract the coordinates from the box
    # clip and normalize here 10 25
    region = x
    # num_iter = 10
    # N = 15

    # for i in range (num_iter):
    # attacks = torch.zeros_like(region).to(device)
    # for n in range(N):
    gauss = torch.randn(region.size()[0], region.size()[1], region.size()[2]) * (sigma / 255)
    gauss = gauss.to(device=device)
    region_dct = dct_2d(region + gauss).to(device=device)
    mask = (torch.rand_like(region) * 2 * rho + 1 - rho).to(device=device)
    region_idct = idct_2d(region_dct * mask)
    region_idct = torch.autograd.Variable(region_idct, requires_grad=True)
    region = clip_by_tensor(region_idct, 0.0, 255.0)
    x = region
    return x
        # region_idct = torch.autograd.Variable(region_idct, requires_grad=True)
        # Calculate SAM loss here
        # region_idct = region_idct.permute(2, 0, 1).contiguous()[None, :, :, :]
        # torch.Size([1, 3, 684, 1024]) (534, 800, 3)
        # print(region_idct.shape, region.shape)
        # attacks += region_idct
    # avg_attack = attacks / N
    # x = avg_attack * 255
    # x = torch.autograd.Variable(region, requires_grad=True)
    # region_idct = idct_2d(region_dct * avg_mask)
    # region_idct = torch.autograd.Variable(region_idct, requires_grad=True)
    # region = clip_by_tensor(region_idct, 0.0, 1.0)

    ############################################

    # print("regionMax: ", region.max(), " ;regionMin: ", region.min())
    # region_idct = torch.clamp(region_idct, 0, 255)
    # region_idct = (region_idct - region_idct.min()) / (region_idct.max() - region_idct.min()) * 255

    # Place the transformed region back into the image
    # x[scaled_miny:scaled_maxy, scaled_minx:scaled_maxx, :] = region_idct
    # x = region * 255
    # print("Max: ", x.max(), " ;Min: ", x.min())
    # TODO: scale back

