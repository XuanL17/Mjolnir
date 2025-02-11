import os
import time

import numpy as np
import torchvision
import torch
import math
import torchvision.transforms as transforms
from torch.optim import Adam,AdamW

from utils.dataset import MnistDataset, TrainDataset
from utils.networkHelper import *

from noisePredictModels.Unet.UNet import Unet
from utils.trainNetworkHelper import SimpleDiffusionTrainer
from models.vision import LeNet, weights_init1

from diffusionModels.simpleDiffusion.simpleDiffusion import DiffusionModel
import matplotlib.pyplot as plt
from torchvision import models, datasets, transforms

from utils.utils import weight_init, channel_deal_3
from util import label_to_onehot, cross_entropy_for_onehot, label_to_onehot2
from sklearn.metrics import accuracy_score
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
plt.xticks([])
plt.yticks([])
plt.axis('off')
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np

def calc_ssim(img1_path, img2_path):
    '''
    Parameters
    ----------
    img1_path : str
        图像1的路径.
    img2_path : str
        图像2的路径.

    Returns
    -------
    ssim_score : numpy.float64
        结构相似性指数（structural similarity index，SSIM）.

    References
    -------
    https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html

    '''
    img1 = Image.open(img1_path).convert('L')
    img2 = Image.open(img2_path).convert('L')
    img2 = img2.resize(img1.size)
    img1, img2 = np.array(img1), np.array(img2)
    # 此处因为转换为灰度值之后的图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    ssim_score = ssim(img1, img2, data_range=255)
    return ssim_score

def gt_psnr(img_batch, ref_batch, batched=False, factor=255):
    """Standard PSNR."""

    def get_psnr(img_in, img_ref):
        mse = ((img_in - img_ref) ** 2).mean()
        if mse > 0 and torch.isfinite(mse):
            return (10 * torch.log10(factor ** 2 / mse))
        elif not torch.isfinite(mse):
            return img_batch.new_tensor(float('nan'))
        else:
            return img_batch.new_tensor(float('inf'))

    if batched:
        psnr = get_psnr(img_batch.detach(), ref_batch)
    else:
        [B, C, m, n] = img_batch.shape
        psnrs = []
        for sample in range(B):
            psnrs.append(get_psnr(img_batch.detach()[sample, :, :, :], ref_batch[sample, :, :, :]))
        psnr = torch.stack(psnrs, dim=0).mean()

    return psnr.item()


def _get_meanstd(dataset):
    cc = torch.cat([dataset[i][0].reshape(1, -1) for i in range(len(dataset))], dim=1)
    data_mean = torch.mean(cc, dim=1).tolist()
    data_std = torch.std(cc, dim=1).tolist()
    return data_mean, data_std


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

torch.multiprocessing.set_start_method('spawn')
"""
-----------------------------------------------------------------------------------------------------------STEP 1: Gradient Diffusion Training
"""

image_size = 120
channels = 1
batch_size = 8

# Uncomment when training
# imagenet_data = TrainDataset("E:\COde\Master\MyDiffusionModels\MyDiffusionModelsGradient\dataset\FashionMNIST_grad2\orig", "E:\COde\Master\MyDiffusionModels\MyDiffusionModelsGradient\dataset\FashionMNIST_grad2\\noise")
#
# data_loader = torch.utils.data.DataLoader(imagenet_data,
#                                           batch_size=batch_size,
#                                           shuffle=True,
#                                           num_workers=0)

# # #
dim_mults = (1, 2, 4,)
denoise_model = Unet(
    dim=120,
    channels=channels,
    dim_mults=dim_mults
)
#
timesteps = 1000
schedule_name = "linear_beta_schedule"
DDPM = DiffusionModel(schedule_name=schedule_name,
                      timesteps=timesteps,
                      beta_start=0.0001,
                      beta_end=0.02,
                      denoise_model=denoise_model).to(device)

optimizer = Adam(DDPM.parameters(), lr=1e-4)
epoches = 20

# Uncomment when training
# Trainer = SimpleDiffusionTrainer(epoches=epoches,
#                                  train_loader=data_loader,
#                                  optimizer=optimizer,
#                                  device=device,
#                                  timesteps=timesteps)

# Training Parameters Setting
root_path = "./saved_train_models/"
setting = "image2Size{}_channels{}_dimMults{}_timeSteps{}_scheduleName{}".format(image_size, channels, dim_mults,
                                                                                 timesteps, schedule_name)

saved_path = os.path.join(root_path, setting)
if not os.path.exists(saved_path):
    os.makedirs(saved_path)

# ---If you want to retrain the model, please uncomment the following line of code.
# Here, I have only trained for 5 epochs, but if you want better results,
# you may need to increase the number of epochs for training. Uncomment when training.
# DDPM = Trainer(DDPM, model_save_path=saved_path)
#
# # #
# # # """
# # # ----------------------------------------------------------------------------------------------------------STEP 2
# # # """
# #
# print('----------------------------------------------------------STEP 2
#
#
# #
best_model_path = saved_path + '/' + 'BestModel.pth'
DDPM.load_state_dict(torch.load(best_model_path,map_location='cuda:0'))

def cal_sensitivity(lr, clip, dataset_size):
    return 2 * lr * clip / dataset_size

def cal_sensitivity_MA(lr, clip, dataset_size):
    return lr * clip / dataset_size

def Gaussian_Simple(epsilon, delta):
    return np.sqrt(2 * np.log(1.25 / delta)) / epsilon

def Laplace(epsilon):
    return 1 / epsilon


#beta,alpha

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):

    return torch.linspace(beta_start, beta_end, timesteps)

betas = linear_beta_schedule(timesteps = 1000)

alphas = 1 - betas

alphas_prod = torch.cumprod(alphas, 0)
#print(alphas_prod)




#DP epsilon can take values of 1, 10, 50, and 100.
epsilon = 10

#delta is generally set to 10e-5, does not need to be changed.
delta = 10e-5

dp_clip = 10

#The attacked LeNet learning rate
lr = 0.001

#gradient inversion attacks, adding noise to a gradient dataset is determined by the number of gradients in a batch.
# If the batch size is 8, then 8 gradients are taken into account and noise is added accordingly.
dp_dataset = 1

####Noise addition

#
# sensitivity
sensitivity = cal_sensitivity(lr, dp_clip, dp_dataset)

#

#Gaussian noise_scale
noise_scale = Gaussian_Simple(epsilon, delta)

# Laplace(epsilon)
# noise_scale = Laplace(epsilon)



#Print
M = sensitivity*noise_scale
print(M)

# Perturbed gradients used for recovery are multiplied by sqrt(n) and directly fed into the trained model.
n = 1/(1+M*M)
n_sample = np.sqrt(n)
print('alphas_prod Minimum n：', n)
print('coefficient to be multiplied during the recovery process n_sqrt：', n_sample)

#The value of t in diffusion denoising step,
# when n is exactly greater than the elements in alphas_prod, is determined.
i = 0
t=0
for t_s in alphas_prod:
    print('{} alpha Multiple_prod {}'.format(i+1,t_s))
    i+=1
    if n > t_s:
        print("when t = {}, just covers the noise required by Gradient Diffusion Model. ".format(i))
        t=i
        break


## Load a specific image and simulate differential privacy encryption by adding Gaussian noise to the image.
dataset = torchvision.datasets.FashionMNIST("./dataset/fashion/raw/",download=True,
                                        transform=transforms.ToTensor())


tt = transforms.ToPILImage()
tp = transforms.ToTensor()


# # -----------------------------------------------------------------------------------------------------------Model loading


true_label = []
image_pred_label = []
text_pred_label = []
fuse_pred_label = []
pred_label=[]
psnr_list=[]
image_loss = []
model = LeNet().to(device)
#
#
torch.manual_seed(0)
torch.cuda.manual_seed(0)
model.apply(weights_init1)

# --------------------------------------------------------------------------------------------------------adjust img_index
org_temp=None
start=time.time()
for index in range(1):
    img_index = 0
    img = dataset[img_index][0].to(device)
    img = img.view(1, *img.size())
    plt.title('Original image')
    plt.imshow(tt(img[0].cpu()))
    plt.show()
    label = torch.Tensor([dataset[img_index][1]]).long().to(device)
    label = label.view(1, )

    # -----------------------------------------------------------------------------------------------------------Gradient Extraction
    loss_fn = nn.CrossEntropyLoss()

    log_probs = model(img)
    loss = loss_fn(log_probs, label)
    grad = torch.autograd.grad(loss, model.parameters())
    or_grad = list((_.detach().clone() for _ in grad))

    te = 0
    for e in or_grad:
        tt1 = np.array(e.cpu())
        tt1 = tt1.flatten()
        tt1 = tt1.tolist()
        or_grad[te] = tt1
        te = te + 1
    channel1 = []
    for layer in range(8):
        channel1.extend(or_grad[layer])
    # Gradient padding
    channel1 = channel_deal_3(channel1)
    channel1 = torch.Tensor(channel1).float().to(device)
    # channel1 = channel1.view(-1)
    channel1 = torch.unsqueeze(channel1, 0)
    print('original Gradient',channel1.shape)
    plt.imshow(tt(channel1.cpu()))
    #
    plt.title('original Gradient')
    plt.show()

    # Noise addition
    channel1 += torch.from_numpy(np.random.normal(loc=0, scale=M,
                                                  size=channel1.shape)).to(device)
    # channel1 += torch.from_numpy(np.random.laplace(loc=0, scale=M,
    #                                                  size=channel1.shape)).to(device)
    channel1=channel1*n_sample

    plt.imshow(tt(channel1.cpu()))

    plt.title('Noise Gradient')
    plt.show()
    noiseimg = channel1.view(1, *channel1.size())



    start=time.time()
    samples = DDPM(mode="generate", noiseimg=noiseimg, image_size=image_size, batch_size=1, channels=channels,recover_t=t)

    generate_image0 = samples[-1][0].reshape(channels, 120, 120)
    figtest0 = tt(torch.from_numpy(generate_image0))
    plt.title('Revoverd Gradient')

    plt.imshow(figtest0)
    plt.show()
    end=time.time()
    print(end-start)
    #     #
    # # --------------------------------------------------------------------------------------Convert the restored gradients back to the original format.
    weights = []
    count = 0
    x_0 = generate_image0.flatten()
    # x_0 = channel1.flatten()
    for name, param in model.named_parameters():
        layer_par = x_0[count:count + param.numel()]
        layer_par = layer_par.reshape(param.shape)
        layer_par=torch.tensor(layer_par)
        weights.append(layer_par.to(device))
        count += param.numel()

#
# #
# #
# #     # # """
# #     # #  -----------------------------------------------------------------------------------------------STEP 3: recover image using the recovered gradients
# #     #
# #     # """
#
    print('----------------------------------------------------------STEP 3: recover image using the recovered gradients')




    dst = datasets.FashionMNIST("~/.torch", download=True)

    dst2 = datasets.FashionMNIST("~/.torch", download=True,transform=transforms.ToTensor())

    dm, ds = _get_meanstd(dst2)
    dm = torch.as_tensor(dm)[:, None, None].to(device)
    ds = torch.as_tensor(ds)[:, None, None].to(device)




    gt_data = tp(dst[img_index][0]).to(device)
    gt_data = gt_data.view(1, *gt_data.size())

    gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label)
    true_label.append(gt_label.item())

    criterion = cross_entropy_for_onehot

    # compute original gradient


    pred = model(gt_data)
    y = criterion(pred, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, model.parameters())

    original_dy_dx = weights



    # generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])
    history = []
    # loss
    temp_loss = 0
    for iters in range(20):

        def closure():
            optimizer.zero_grad()

            dummy_pred = model(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()

            return grad_diff


        optimizer.step(closure)
        if iters % 10 == 0:
            current_loss = closure()
            temp_loss = current_loss
            print(iters, "%.4f" % current_loss.item())
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.imshow(tt(dummy_data[0].cpu()))
            plt.show()

            history.append(tt(dummy_data[0].cpu()))

    image_loss.append(temp_loss.item())
    recoverd_number = [i for i in image_loss if i < 50]
    with torch.no_grad():
        print(gt_data.squeeze(0).shape)
        SSIM1=ssim(gt_data.squeeze(0).cpu().numpy(), dummy_data.squeeze(0).cpu().numpy(),data_range=255, channel_axis=0)
        MSE1=mse(gt_data.squeeze(0).cpu().numpy(), dummy_data.squeeze(0).cpu().numpy())
        psnr = gt_psnr(gt_data, dummy_data)
        print('-----------------:', psnr)
        if psnr > 0:
            psnr_list.append(psnr)
        else:
            psnr_list.append(0)
    pred_label.append(torch.argmax(dummy_label).item())
    print('temp loss:', image_loss)
    print('true label:', true_label)
    print('pred label:', pred_label)
    print('label acc:', accuracy_score(true_label, pred_label))
    print('image recu acc:', len(recoverd_number) / len(image_loss))
    print('avg psnr:', np.mean(psnr_list))
    print(SSIM1)
    print(MSE1)

