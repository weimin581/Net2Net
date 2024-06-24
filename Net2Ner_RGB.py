import os
from threading import Thread  # for running the denoiser in parallel
import queue  # 队列

import numpy as np
import torch
import torch.optim
from models.skip import skip  # our network

from utils.utils import *  # auxiliary functions

from utils.data import Data  # class that holds img, psnr, time

from skimage.restoration import denoise_nl_means
from dncnn_models.network_dncnn import DnCNN as net # dncnn net
from dncnn_models.network_ffdnet import FFDNet as net_ffdnet    # FFDNet

# repalce ---> FFDNet ---> DRUNet
from models_drunet.network_unet import UNetRes as net

from utils_drunet import utils_logger
from utils_drunet import utils_model
from utils_drunet import utils_image as util

import warnings
warnings.filterwarnings("ignore")

CUDA_FLAG = True
CUDNN = True
if CUDA_FLAG:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # GPU accelerated functionality for common operations in deep neural nets
    torch.backends.cudnn.enabled = CUDNN
    
    torch.backends.cudnn.benchmark = CUDNN
    # torch.backends.cudnn.deterministic = True
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

X_LABELS = ['Iterations']*3
Y_LABELS = ['PSNR between x and net (db)', 'PSNR with original image (db)', 'loss']
ORIGINAL = 'Clean'
CORRUPTED = 'Noisy'


def load_image(fclean_path, sigma):

    _, img_np = load_and_crop_image(fclean_path)

    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma / 255., size=img_np.shape), 0, 1).astype(np.float32)

    data_dict = {ORIGINAL: Data(img_np), CORRUPTED: Data(img_noisy_np, compare_psnr(img_np, img_noisy_np))}

    return data_dict    


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")


model.load_state_dict(torch.load(model_path), strict=True)
model.eval()

for k, v in model.named_parameters():
    v.requires_grad = False

model = model.to(device)

def DRUNet_rgb_yuan(noisy_np_img,sigma):

    # 预处理：
    noisy_np_img = np.transpose(noisy_np_img,(1,2,0))
    # print(noisy_np_img.shape)
    img_L = util.single2tensor4(noisy_np_img)
    img_L = torch.cat((img_L, torch.FloatTensor([sigma/255.]).repeat(1, 1, img_L.shape[2], img_L.shape[3])), dim=1)
    img_L = img_L.to(device)
    # print(img_L.shape) # torch.Size([1, 2, 256, 256]) 
    


    img_E = utils_model.test_mode(model, img_L, refield=64, mode=5)  # 执行
    img_E = img_E.cpu()
         
    return np.array(img_E, dtype=np.float32)     

def train_via_admm(net, net_input, denoiser_function, y, org_img=None,                      # y is the noisy image
                   admm_iter=3000, save_path="",           # path to save params
                   LR=0.008,                                     # learning rate
                   sigma_f=3, update_iter=10, method='fixed_point',   # method: 'fixed_point' or 'grad' or 'mixed'
                   beta=.5, mu=.5, LR_x=None, noise_factor=0.033,        # LR_x needed only if method!=fixed_point
                   threshold=20, threshold_step=0.01, increase_reg=0.03):                # increase regularization 
    
    # get optimizer and loss function:
    mse = torch.nn.MSELoss().type(dtype)  # using MSE loss
    # additional noise added to the input:
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    if org_img is not None:
        psnr_y = compare_psnr(org_img, y)  # get the noisy image psnr
        
    # x update method:
    if method == 'fixed_point':
        swap_iter = admm_iter + 1
        LR_x = None
    elif method == 'grad':
        swap_iter = -1
    elif method == 'mixed':
        swap_iter = admm_iter // 2
    else:
        assert False, "method can be 'fixed_point' or 'grad' or 'mixed' only "
    
    # optimizer and scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)  # using ADAM opt
    y_torch = np_to_torch(y).type(dtype)
    x = y.copy()
    u = np.zeros_like(y)
    f_x = x.copy()
    avg = np.rint(y)

    psnr_net_list=[]
    psnr_x_list=[]
    psnr_x_u_list=[]
    psnr_avg_list=[]
    image_list  = []

    for i in range(1, 1 + admm_iter):
        # step 1, update network:
        optimizer.zero_grad()
        net_input = net_input_saved + (noise.normal_() * noise_factor)
        
        out = net(net_input)         #原始DIP结果
        out_np = torch_to_np(out)    #转化numpy为了计算psnr
        # loss:
        loss_y = mse(out, y_torch)
        loss_x = mse(out, np_to_torch(x - u).type(dtype))  
        total_loss = loss_y + mu * loss_x                  # 新的Loss
        total_loss.backward()
        optimizer.step()

        # step 2, update x using a denoiser and result from step 1
        f_x = denoiser_function(x.copy(), sigma_f)
        
        # 使用深度先验的话需要去掉一维：
        f_x=np.squeeze(f_x)

        if i < swap_iter:
            x = 1 / (beta + mu) * (beta * f_x + mu * (out_np + u))
        else:
            x = x - LR_x * (beta * (x - f_x) + mu * (x - out_np - u))

        np.clip(x, 0, 1, out=x)  # making sure that image is in bounds

        # step 3, update u
        u = u + out_np - x  

        # Averaging: 等同于DIP
        avg = avg * .99 + out_np * .01

        # show psnrs:
        psnr_noisy = compare_psnr(out_np, y)
        if psnr_noisy > threshold:
            mu = mu + increase_reg
            beta = beta + increase_reg
            threshold += threshold_step

        if org_img is not None:
            psnr_net, psnr_avg = compare_psnr(org_img, out_np), compare_psnr(org_img, avg)
            psnr_x, psnr_x_u = compare_psnr(org_img, x), compare_psnr(org_img, x - u)

            psnr_avg_list.append(psnr_avg)
            image_list.append(avg)
            psnr_max_temp=max(psnr_avg_list)                       
            psnr_max_temp_index=psnr_avg_list.index(psnr_max_temp)  
            print('\r', '%04d/%04d Loss %f' % (i, admm_iter, total_loss.item()),'初始PSNR: %.2f  目前最佳PSNR: %.2f 目前最佳PSNR的迭代数: %d' % (psnr_y,psnr_max_temp,psnr_max_temp_index), end='')
            
            #记录4组psnr:
            psnr_net_list.append(psnr_net)
            psnr_x_list.append(psnr_x)
            psnr_x_u_list.append(psnr_x_u)
            

        else:
            print('\r', 'iteration %04d/%04d Loss %f' % (i, admm_iter, total_loss.item()), end='')
    
    return avg,psnr_net_list,psnr_x_list,psnr_x_u_list,psnr_avg_list


