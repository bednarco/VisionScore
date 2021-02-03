import os

import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm_notebook as tqdm

from sportsfield_release.utils import utils, warp, image_utils, constant_var
from sportsfield_release.models import end_2_end_optimization
from sportsfield_release.options import fake_options
import cv2

# if want to run on CPU, please make it False
constant_var.USE_CUDA = False
utils.fix_randomness()

# if GPU is RTX 20XX, please disable cudnn
torch.backends.cudnn.enabled = True

# set some options
opt = fake_options.FakeOptions()
opt.batch_size = 1
opt.coord_conv_template = True
opt.error_model = 'loss_surface'
opt.error_target = 'iou_whole'
# opt.goal_image_path = './clips/frame0.jpg'
opt.guess_model = 'init_guess'
opt.homo_param_method = 'deep_homography'
opt.load_weights_error_model = 'pretrained_loss_surface'
opt.load_weights_upstream = 'pretrained_init_guess'
opt.lr_optim = 1e-5
opt.need_single_image_normalization = True
opt.need_spectral_norm_error_model = True
opt.need_spectral_norm_upstream = False
opt.optim_criterion = 'l1loss'
opt.optim_iters = 200
opt.optim_method = 'stn'
opt.optim_type = 'adam'
opt.out_dir = './out'
opt.prevent_neg = 'sigmoid'
opt.template_path = '../data/template.png'
opt.warp_dim = 8
opt.warp_type = 'homography'

def read_template():
    template_image_np = imageio.imread(opt.template_path, pilmode='RGB')
    template_image_np = template_image_np / 255.0
    # template_image_np = image_utils.rgb_template_to_coord_conv_template(template_image_np)
    template_image_torch = utils.np_img_to_torch_img(template_image_np)
    return template_image_np, template_image_torch

def show_field(temp, img, h):
    template_image_draw = temp
    goal_image_draw = img / 255.0
    outshape = goal_image_draw.shape[0:2]    
    optim_homography = h
    warped_tmp_optim = warp.warp_image(template_image_draw, optim_homography, out_shape=outshape)[0]
    warped_tmp_optim = utils.torch_img_to_np_img(warped_tmp_optim)
    return warped_tmp_optim

def show_top_down(temp, img, h):
    H_inv = torch.inverse(h)
    template_image_draw = temp
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    goal_image_draw = img / 255.0
    outshape = template_image_draw.shape[1:3]
    warped_frm = warp.warp_image(utils.np_img_to_torch_img(goal_image_draw)[None], H_inv, out_shape=outshape)[0]
    warped_frm = utils.torch_img_to_np_img(warped_frm)+utils.torch_img_to_np_img(template_image_draw)*0.5
    return warped_frm

def transform(h):
    out = warp.get_four_corners(h, utils.to_torch(utils.FULL_CANON4PTS_NP()))
    out = out.permute(0, 2, 1)
    print(out*2)

# def show_top_down2(temp, img, h):
#     H_inv = torch.inverse(h)
#     template_image_draw = temp
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     goal_image_draw = img / 255.0
#     outshape = template_image_draw.shape[1:3]
#     warped_frm = warp.warp_image(utils.np_img_to_torch_img(goal_image_draw)[None], H_inv, out_shape=outshape)[0]
#     warped_frm = utils.torch_img_to_np_img(warped_frm)
#     t = utils.torch_img_to_np_img(template_image_draw)
#     t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
#     # print(cv2.findNonZero(warped_frm))
#     for point in cv2.findNonZero(warped_frm):
#         print (point[0])
#         cv2.circle(t, (point[0][0],point[0][1]), 8, (255,0,0), -1)
#     # print(np.transpose(np.nonzero(warped_frm)))
#     return t