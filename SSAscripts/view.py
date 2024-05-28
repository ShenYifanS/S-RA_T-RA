import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

neg_th = -10


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


def main():
    device = "cpu"
    map_location = torch.device('cpu')
    image = cv2.imread('./keji.png')
    # adv_image = torch.load('./keji.png_vit_b_epsilon_16.0.pt')
    adv_image = torch.load('./keji.png_vit_b_epsilon_16.0.pt', map_location='cpu').to(device=device)
    adv_image_np = adv_image[0].permute(1, 2, 0).detach().cpu().numpy()
    adv_image_np = cv2.resize(adv_image_np.astype(np.uint8),
                           (image.shape[1], image.shape[0]))
    # cv2.imshow('img', adv_image_np)
    # cv2.waitKey(5000)
    sam_checkpoint = "./../checkpoints/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    # input_point = np.array([[500, 375]])
    # input_point = np.array([[516, 303],[116,103],[616, 303],[416, 303],[380,260],[650,103]])
    input_point = np.array([[500, 103]])
    input_label = np.array([1])
    # predictor.set_image(image)
    predictor.set_torch_image(adv_image, adv_image_np.shape[:2])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
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
        plt.savefig(f'output_{i}.png')
        # plt.show()
    # plt.figure(figsize=(10, 10))
    # plt.imshow(adv_image_np)
    # show_points(input_point, input_label, plt.gca())
    # plt.savefig('proposal_figures/orig_prompt.pdf')


if __name__ == '__main__':
    main()
