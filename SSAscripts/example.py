import numpy as np
import torch
# import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

from functools import partial
from typing import Optional, Tuple, Union
# from adv_lib.utils.losses import difference_of_logits, difference_of_logits_ratio
from torch import Tensor, nn
from torch.autograd import grad, Variable
from torch.nn import functional as F


# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def pgd(predictor, image, input_point, input_label, device, ε, restarts=1,
         neg_th=-10, num_steps=40, relative_step_size=2./255):
    torch_image = torch.as_tensor(image, dtype=torch.float)
    if isinstance(ε, (int, float)):
        ε = torch.full((len(torch_image),), ε, dtype=torch.float, device=device)

    adv_image = None
    best_loss = float('inf')

    for i in range(restarts):
        adv_loss_run, adv_image_run = _pgd(predictor=predictor, image=image, input_point=input_point,
                      input_label=input_label, device=device, ε=ε, neg_th=neg_th,
                      num_steps=num_steps, relative_step_size=relative_step_size, random_init=(i!=0))
        if adv_loss_run <= best_loss:
            best_loss = adv_loss_run
            adv_image = adv_image_run

    return adv_image


def _pgd(predictor, image, input_point, input_label, device, ε,
         neg_th=-10, num_steps=10, relative_step_size=2./255, random_init=False):
    loss_function = nn.MSELoss()
    input_image = predictor.transform.apply_image(image)
    torch_image = torch.as_tensor(input_image, dtype=torch.float32, device=device)
    # torch_image = Variable(torch_image, requires_grad=True)
    batch_view = lambda tensor: tensor.view(1, *[1] * (torch_image.ndim - 1))
    neg_inputs = -torch_image
    one_minus_inputs = 255. - torch_image

    step_size = ε * relative_step_size

    δ = torch.zeros_like(torch_image, requires_grad=True)
    best_loss = float('inf')
    best_adv = torch_image.clone()

    if random_init:
        δ.data.uniform_().sub_(0.5).mul_(2 * batch_view(ε))

    for i in range(num_steps):
        adv_image = torch_image + δ
        adv_image = adv_image.permute(2, 0, 1).contiguous()[None, :, :, :]
        predictor.set_torch_image(adv_image, image.shape[:2])
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
            return_logits=True
        )
        # masks: N, W, H, torch

        # if return logits, masks are logits
        # logits, 1, 256, 256
        # scores correspond to mask, list of scores
        neg_threshold = torch.full(masks.size(), neg_th, dtype=torch.float32)
        # neg_threshold = torch.zeros_like(logits, requires_grad=True).fill_(neg_th)
        loss = loss_function(torch.clamp(masks, min=neg_th, max=None), neg_threshold)
        print(loss)
        δ_grad = - grad(loss.sum(), δ, only_inputs=True)[0]
        if loss.item() <= best_loss:
            best_loss = loss.item()
            best_adv = adv_image
            torch.save(adv_image, 'adv_image.pt')
        δ.data.add_(batch_view(step_size) * δ_grad.sign()).clamp_(min=batch_view(-ε), max=batch_view(ε))
        δ.data.clamp_(min=neg_inputs, max=one_minus_inputs)

    return best_loss, best_adv


def minimal_pgd(predictor, image, input_point, input_label, device,
                max_ε, binary_search_steps=20, adv_threshold=0.3):

    adv_image = image.copy()
    best_ε = torch.full((1,), 2 * max_ε, dtype=torch.float, device=device)
    ε_low = torch.zeros_like(best_ε)

    for i in range(binary_search_steps):
        ε = (ε_low + best_ε) / 2
        print('pgd trying ε =', ε)
        adv_image_run = pgd(predictor=predictor, image=image, input_point=input_point,
                            input_label=input_label, device=device, ε=ε,num_steps=5)
        # check attack results
        predictor.set_torch_image(adv_image_run, image.shape[:2])
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            # multimask_output=False,
            multimask_output=True,
        )
        # masks: N, W, H
        # mask_score: N,
        adv_percent = scores.flatten(1).sum(dim=1) / masks.flatten(1).sum(dim=1)
        better_adv = (adv_percent >= adv_threshold) & (ε < best_ε)
        adv_image[better_adv] = adv_image_run[better_adv]

        ε_low = torch.where(better_adv, ε_low, ε)
        best_ε = torch.where(better_adv, ε, best_ε)

    return adv_image

def main():
    image = cv2.imread('notebooks/images/truck.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    input_point = np.array([[500, 375]])# point position
    input_label = np.array([1])

    adv_image = minimal_pgd(predictor=predictor, image=image, input_point=input_point,
                            input_label=input_label, device=device, max_ε=8.)

if __name__ == '__main__':
    main()

# 区域选择, 区域攻击，sample point，攻击效果 