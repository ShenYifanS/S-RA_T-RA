import argparse
import csv
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append(".")
from segment_anything import sam_model_registry, SamPredictor
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
import scipy.spatial.distance as dist

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
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def calculate_iou(mask1, mask2):
    mask1_flat = mask1.flatten()
    mask2_flat = mask2.flatten()
    iou = jaccard_score(mask1_flat, mask2_flat)
    return iou


# def calculate_f1(mask1, mask2):
#     mask1_flat = mask1.flatten()
#     mask2_flat = mask2.flatten()
#     f1 = f1_score(mask1_flat, mask2_flat)
#     return f1
#
# def calculate_dice(mask1, mask2):
#     mask1_flat = mask1.flatten()
#     mask2_flat = mask2.flatten()
#     intersection = np.logical_and(mask1_flat, mask2_flat).sum()
#     return 2. * intersection / (mask1_flat.sum() + mask2_flat.sum())
#
# def calculate_hausdorff(mask1, mask2):
#     # 找出两个掩模的边缘点
#     points_mask1 = np.transpose(np.nonzero(mask1))
#     points_mask2 = np.transpose(np.nonzero(mask2))
#
#     # 计算所有点对之间的距离
#     hausdorff_dist = dist.directed_hausdorff(points_mask1, points_mask2)[0]
#     return hausdorff_dist


def maineval(args):
    sam_checkpoint = args.sam_checkpoint
    pdf_filename = args.pdf_filename
    with PdfPages(pdf_filename) as pdf:

        # for point in ([args.input_point[0], args.input_point[1]],[args.input_point[0]-30, args.input_point[1]+30],[args.input_point[0]+30, args.input_point[1]-30]):
        for point in ([args.input_point]):
            input_point = np.array([point])

            image = cv2.imread(args.image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            adv_image = torch.load(args.adv_image_path, map_location='cpu').to(device=args.device)
            adv_image_np = adv_image[0].permute(1, 2, 0).detach().cpu().numpy()
            adv_image_np = cv2.resize(adv_image_np.astype(np.uint8), (image.shape[1], image.shape[0]))


            sam = sam_model_registry[args.model_type](checkpoint=sam_checkpoint)
            sam.to(device=args.device)
            predictor = SamPredictor(sam)

            # input_point = np.array([args.input_point])
            input_label = np.array([1])
            # input_point = torch.tensor(input_point, device=args.device)
            # input_label = torch.tensor(input_label, device=args.device)

            # 对原始图像进行分割
            predictor.set_image(image)
            original_masks, _a, _b = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            areas = np.sum(original_masks, axis=(1, 2))
            sorted_indices = np.argsort(areas)
            original_masks = original_masks[sorted_indices]
            _a = _a[sorted_indices]
            _b = _b[sorted_indices]

            predictor.set_torch_image(adv_image, adv_image_np.shape[:2])
            adv_masks, adv_scores, _c = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )

            areas = np.sum(adv_masks, axis=(1, 2))
            sorted_indices = np.argsort(areas)
            adv_masks = adv_masks[sorted_indices]
            adv_scores = adv_scores[sorted_indices]
            _c = _c[sorted_indices]


            # 输出原始图像和掩模
            plt.figure(figsize=(10, 10))
            plt.imshow(image_rgb)
            show_mask(original_masks[0], plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.title(f"Original Image and Mask", fontsize=18)
            plt.axis('off')
            pdf.savefig()
            plt.close()

            iou_score = calculate_iou(original_masks[0], adv_masks[0])
            # f1_score = calculate_f1(original_masks[0], adv_masks[0])
            # dice_score = calculate_dice(original_masks[0], adv_masks[0])
            # hausdorff_distance = calculate_hausdorff(original_masks[0], adv_masks[0])
            plt.figure(figsize=(10, 10))
            plt.imshow(adv_image_np)
            show_mask(adv_masks[0], plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.title(
                f"Adv Mask, Score: {adv_scores[0]:.3f}, IoU: {iou_score:.3f}",
                fontsize=16)
            with open(args.result_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                # writer.writerow(['name', 'IoU', 'F1', 'Dice', 'Hausdorff'])
                writer.writerow([args.adv_image_path, iou_score])
            plt.axis('off')
            pdf.savefig()
            plt.close()

            plt.figure(figsize=(10, 10))
            plt.imshow(image_rgb)
            show_mask(original_masks[1], plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.title(f"Original Image and Mask", fontsize=18)
            plt.axis('off')
            pdf.savefig()
            plt.close()

            iou_score = calculate_iou(original_masks[1], adv_masks[1])
            # f1_score = calculate_f1(original_masks[1], adv_masks[1])
            # dice_score = calculate_dice(original_masks[1], adv_masks[1])
            # hausdorff_distance = calculate_hausdorff(original_masks[1], adv_masks[1])
            plt.figure(figsize=(10, 10))
            plt.imshow(adv_image_np)
            show_mask(adv_masks[1], plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.title(
                f"Adv Mask, Score: {adv_scores[1]:.3f}, IoU: {iou_score:.3f}",
                fontsize=16)
            with open(args.result_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                # writer.writerow(['name', 'IoU', 'F1', 'Dice', 'Hausdorff'])
                writer.writerow([args.adv_image_path, iou_score])

            plt.axis('off')
            pdf.savefig()
            plt.close()

            plt.figure(figsize=(10, 10))
            plt.imshow(image_rgb)
            show_mask(original_masks[2], plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.title(f"Original Image and Mask", fontsize=18)
            plt.axis('off')
            pdf.savefig()
            plt.close()

            iou_score = calculate_iou(original_masks[2], adv_masks[2])
            # f1_score = calculate_f1(original_masks[2], adv_masks[2])
            # dice_score = calculate_dice(original_masks[2], adv_masks[2])
            # hausdorff_distance = calculate_hausdorff(original_masks[2], adv_masks[2])
            plt.figure(figsize=(10, 10))
            plt.imshow(adv_image_np)
            show_mask(adv_masks[2], plt.gca())
            show_points(input_point, input_label, plt.gca())
            plt.title(
                f"Adv Mask, Score: {adv_scores[2]:.3f}, IoU: {iou_score:.3f}",
                fontsize=16)
            with open(args.result_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                # writer.writerow(['name', 'IoU', 'F1', 'Dice', 'Hausdorff'])
                writer.writerow([args.adv_image_path, iou_score])
            plt.axis('off')
            pdf.savefig()
            plt.close()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--image_path', type=str, default='./keji.png', help='Path to the input image')
#     parser.add_argument('--adv_image_path', type=str, default='./keji.png_vit_b_epsilon_16.0.pt', help='Path to the adversarial image')
#     parser.add_argument('--sam_checkpoint', type=str, default="./../checkpoints/sam_vit_l_0b3195.pth", help='Path to the SAM checkpoint file')
#     parser.add_argument('--input_point', nargs=2, type=int, default=[[516, 303],[116,103]], help='Input point coordinates')
#     parser.add_argument('--adv', type=str, default="origin", help='Prefix for the output file paths')
#     # parser.add_argument('--pdf_filename', type=str, default="./X.pdf")
#     args = parser.parse_args()
#     maineval(args)
