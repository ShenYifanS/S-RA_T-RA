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

import matplotlib.pyplot as plt
from segment_anything.utils_ import *

# def show_box(box, ax):
#     x0, y0 = box[0], box[1]
#     w, h = box[2] - box[0], box[3] - box[1]
#     ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

import numpy as np
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

def box_to_samples(box: tuple, x_samples = None, y_samples = None):
    # box:(minx, maxx, miny, maxy)
    # image_size: (H,W)
    # output: (N_point,1,2)
    
    minx, maxx, miny, maxy = box
    assert minx < maxx, "the box should have positive volume"
    assert miny < maxy, "the box should have positive volume"
    if x_samples is None:
        x_samples = 5
    if y_samples is None:
        y_samples = 5
        
    xs = np.round(np.linspace(minx, maxx, x_samples), decimals=0).astype(int)
    ys = np.round(np.linspace(miny, maxy, y_samples), decimals=0).astype(int)
    samples = []
    for x in xs:
        for y in ys:
            samples.append([x,y])
    samples = np.array(samples)[:,None]
    # N_points, 2
    return samples


ar = box_to_samples((0,50,0,50))

def pgd(predictor, image, input_points, input_label, device, ε, restarts=1,
         neg_th=-10, num_steps=2, relative_step_size=2./255):
    torch_image = torch.as_tensor(image, dtype=torch.float)
    if isinstance(ε, (int, float)):
        ε = torch.full((len(torch_image),), ε, dtype=torch.float, device=device)

    adv_image = None
    best_loss = float('inf')

    for i in range(restarts):
        adv_loss_run, adv_image_run = _pgd(predictor=predictor, image=image, input_points=input_points,
                      input_label=input_label, device=device, ε=ε, neg_th=neg_th,
                      num_steps=num_steps, relative_step_size=relative_step_size, random_init=(i!=0))
        if adv_loss_run <= best_loss:
            best_loss = adv_loss_run
            adv_image = adv_image_run

    return adv_image


def _pgd(predictor, image, input_points, input_label, device, ε,
         neg_th=-10, num_steps=20, relative_step_size=2./255, random_init=False):
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
        total_loss = 0
        for input_point in input_points:
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
                return_logits=True
            )
            neg_threshold = torch.full(masks.size(), neg_th, dtype=torch.float32)
            # neg_threshold = torch.zeros_like(logits, requires_grad=True).fill_(neg_th)
            loss = loss_function(torch.clamp(masks, min=neg_th, max=None), neg_threshold)
            total_loss += loss
        adv_loss = total_loss / input_points.shape[0]
        print(f'loss at step {i}: {adv_loss}')
        δ_grad = - grad(adv_loss.sum(), δ, only_inputs=True)[0]
        if loss.item() <= best_loss:
            best_loss = loss.item()
            best_adv = adv_image
            # torch.save(adv_image, 'adv_image_multi.pt')
        print(step_size.size(), ε.size())
        δ.data.add_(batch_view(step_size) * δ_grad.sign()).clamp_(min=batch_view(-ε), max=batch_view(ε))
        δ.data.clamp_(min=neg_inputs, max=one_minus_inputs)

    return best_loss, best_adv

import time
def minimal_pgd(predictor, image, input_points, input_label, device,
                max_ε, binary_search_steps=5, adv_threshold=0.3):

    adv_image = image.copy()
    best_ε = torch.full((1,), 2 * max_ε, dtype=torch.float, device=device)
    ε_low = torch.zeros_like(best_ε)
    image_size = (image.shape[1], image.shape[0])

    orig_areas = []
    predictor.set_image(image)
    for input_point in input_points:
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        masked_areas = masks.sum(axis=-1).sum(axis=-1) / masks.size
        np.sort(masked_areas)
        orig_areas.append(masked_areas)
    # save original image

    for i in range(binary_search_steps):
        start_time = time.time()
        ε = (ε_low + best_ε) / 2
        print('pgd trying ε =', ε)
        total_adv_percent = 0
        adv_image_run = pgd(predictor=predictor, image=image, input_points=input_points,
                            input_label=input_label, device=device, ε=ε)

        # 1, 3, 683, 1024
        print(f'saving adv image at binary search step: {i}')
        # save as jpg
        adv_image_np = adv_image_run[0].permute(1, 2, 0).detach().cpu().numpy()
        adv_image_np = cv2.resize(adv_image_np.astype(np.uint8),
                            image_size)
        plt.clf()
        plt.axis('off')
        plt.imshow(adv_image_np)
        plt.savefig(f'proposal_figures_v2/adv_region_step_{i}_epsilon_{ε.item()}.jpg')  

        # show difference
        # H, W, 3
        input_image = predictor.transform.apply_image(image)
        # 683, 1024, 3
        input_image_torch = torch.as_tensor(input_image, dtype=torch.float32, device=device)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        # 683, 1024, 3
        difference = (adv_image_run - input_image_torch)[0].permute(1, 2, 0).detach().cpu().numpy()
        # range: +-16/255

        difference_image =128* (difference - difference.min())/(difference.max() - difference.min())
        difference_image = cv2.resize(difference_image.astype(np.uint8),
                            (image.shape[1], image.shape[0]))

        plt.clf()
        plt.axis('off')
        plt.imshow(difference_image)
        plt.savefig(f'proposal_figures_v2/difference_step_{i}_epsilon_{ε.item()}.jpg')  
        # visualize evaluation
        evaluation(predictor, image, adv_image_run, adv_image_np, image_size, i)

        # evaluation
        predictor.set_torch_image(adv_image_run, image.shape[:2])
        for j in range(input_points.shape[0]):
            input_point = input_points[i]
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            masked_areas = masks.sum(axis=-1).sum(axis=-1) / masks.size
            np.sort(masked_areas)
            adv_percent = np.mean(masked_areas / orig_areas[j])
            total_adv_percent += adv_percent
        avg_adv_percent = total_adv_percent / input_points.shape[0]
        print(f'adv percent: {avg_adv_percent}')
        if avg_adv_percent <= adv_threshold:
            best_ε = ε
            adv_image = adv_image_run
        else:
            ε_low = ε
        end_time = time.time()
        print(f"Iteration {i}, time consumption: {int(end_time-start_time)}s")
    # torch.save(adv_image, 'adv_image_multi.pt')
    return adv_image

def main():
    # clear previous data
    delete_all_files_in_directory('/home/ec2-user/Attack-SAM/proposal_figures_v2')
    image = cv2.imread('notebooks/images/truck.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_checkpoint = "checkpoints/sam_vit_b_01ec64.pth"
    model_type = "vit_b"
    device = "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    # box_samples = box_to_samples(())
    input_points = np.array([[[500, 375]], [[450, 375]], [[500, 355]],
                            [[500, 395]], [[450, 355]], [[450, 395]]])
    input_label = np.array([1])

    adv_image = minimal_pgd(predictor=predictor, image=image, input_points=input_points,
                            input_label=input_label, device=device, max_ε=8.)

def evaluation(predictor, image, adv_image, adv_image_np, img_size: tuple, step: int):
    # given image, return
    input_point = np.array([[500, 375]])
    input_label = np.array([1])

    # before attack
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(adv_image_np)
        # plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(f'proposal_figures_v2/gt_step_{step}_test_{i}.jpg') 

    #
    plt.clf()
    
    predictor.set_torch_image(adv_image, img_size)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
        # return_logits=True,
    )
    # neg_threshold = torch.full(masks.size(), neg_th, dtype=torch.float32)
    # loss_function = nn.MSELoss()
    # loss = loss_function(torch.clamp(masks, min=neg_th, max=None), neg_threshold)
    # print(loss)
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(adv_image_np)
        # plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(f'proposal_figures_v2/adv_step_{step}_test_{i}.jpg')

if __name__ == '__main__':
    main()
