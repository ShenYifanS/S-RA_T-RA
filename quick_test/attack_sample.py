import os
import cv2
import numpy as np
import torch
import sys
# from parser import args
sys.path.append(".")
from segment_anything import sam_model_registry, SamPredictor
sys.path.append("..")
from SSAscripts.pgd_attack import pgd
import argparse
import matplotlib.pyplot as plt


def box_to_samples(box: tuple, mi, x_samples=1, y_samples=1):
    # box:(minx, maxx, miny, maxy)
    # image_size: (H,W)
    # output: (N_point,1,2)

    minx, maxx, miny, maxy = box
    assert minx < maxx, "the box should have positive volume"
    assert miny < maxy, "the box should have positive volume"
    # x_samples = (maxx - minx)//mi
    # y_samples = (maxy - miny) // mi
    if x_samples is None:
        x_samples = 5
    if y_samples is None:
        y_samples = 5
    print("x_samples: ", x_samples, "  y_samples: ", y_samples)
    xs = np.round(np.linspace(minx, maxx, x_samples), decimals=0).astype(int)
    ys = np.round(np.linspace(miny, maxy, y_samples), decimals=0).astype(int)
    samples = []
    for x in xs:
        for y in ys:
            samples.append([x, y])
    samples = np.array(samples)[:, None]
    # N_points, 2
    return samples

from quick_test.plot_utils import show_points, show_box, show_sampled_points, show_test_points
from quick_test.test_utils import run_test
def main(args):
    # clear previous data
    # delete_all_files_in_directory('/home/ec2-user/Attack-SAM/proposal_figures_v2')
    ori_image = cv2.imread(args.orig_image)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    sam_checkpoint = args.checkpoint
    model_type = args.sam_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    box_samples = box_to_samples(args.box, args.mi, x_samples=5, y_samples=5)
    input_label = np.array([1])

    height, width, _ = image.shape

    # ori image
    if not os.path.exists('sample/'):
        os.mkdir('sample/')

    plt.clf()
    dpi = 100
    figsize = (width / 100, height / 100)  # 你可以根据需要调整缩放比例
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.axis('off')
    ax.imshow(image)
    fig.savefig(f'sample/ori_image.png', dpi=dpi, bbox_inches='tight', pad_inches=0)

    # plot box
    print('original image size')
    print(image.shape)
    plt.clf()
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)
    ax.axis('off')
    show_box(args.box, ax)

    # show box
    fig.savefig(f'sample/box_image.png', dpi=dpi, bbox_inches='tight', pad_inches=0)

    # sampled points
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    ax.imshow(image)
    show_sampled_points(box_samples[:,0], ax)
    # show box
    fig.savefig(f'sample/sampled_points.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    run_test(np.array(args.input_point)[None], input_label, predictor, image, 'sample/', device, phase='before')

    adv_image = pgd(args, predictor=predictor, image=image, input_points=box_samples,
                            input_label=input_label, device=device, ε=args.epsilon, apply_ssa=args.apply_ssa, num_steps=20)# 20
    # print adv image
    adv_image = adv_image.cpu().detach().numpy()
    adv_image_np = np.transpose(adv_image[0], (1,2,0)) # 690, 1024, 3
    adv_image_np = cv2.resize(adv_image_np, (image.shape[1], image.shape[0])).astype(np.uint8)
    assert adv_image_np.max()<=255
    # resize_back?
    # TO be verified
    plt.clf()
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    ax.imshow(image)
    show_test_points(np.array(args.input_point)[None], ax)
    fig.savefig(f'sample/test_point.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    # show box
    
    run_test(np.array(args.input_point)[None], input_label, predictor, adv_image_np, 'sample/', device, phase='after')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--orig_image', type=str, default='testimage/horse.png')
    parser.add_argument('--sam_model', type=str, default='vit_b',
                        help='[vit_b, vit_h]')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                        help='[sam_vit_l_0b3195.pth, sam_vit_b_01ec64.pth, sam_vit_h_4b8939.pth]')
    parser.add_argument('--box', type=lambda s: tuple(map(int, s.split(','))), default=(396.0,538,262,444),
                        help='(minx, maxx, miny, maxy)')
    parser.add_argument('--epsilon', type=float, default=16., help='2, 4, 8, 16')
    # parser.add_argument('--apply_ssa', type=bool, default=True, help='True or False')
    parser.add_argument('--apply_ssa', action='store_true', help='Enable SSA')

    # parser.add_argument('--image_path', type=str, default='testimage/bear.png', help='Path to the input image')
    parser.add_argument('--adv_image_path', type=str, default='', help='Path to the adversarial image')
    parser.add_argument('--sam_checkpoint', type=str, default="checkpoints/sam_vit_b_01ec64.pth", help='Path to the SAM checkpoint file')
    parser.add_argument('--input_point', nargs=2, type=int, default=[446,320], help='Input point coordinates') # [446,320]
    parser.add_argument('--adv', type=str, default="origin", help='Prefix for the output file paths')
    parser.add_argument('--result_csv', type=str, default='SSAscripts/B_B_True_8.csv')
    parser.add_argument('--rho', type=float, default=0.1)
    parser.add_argument('--mi', type=int, default=-1)
    args = parser.parse_args()
    print(args)
    ##################
    args.savetorch = f"result/{args.sam_model}_{args.apply_ssa}_{args.epsilon}_{args.rho}_{args.mi}/{args.orig_image}_{args.sam_model}_epsilon_{args.epsilon}.pt"
    main(args)
