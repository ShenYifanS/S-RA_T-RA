import os
import cv2
import numpy as np
import torch
import sys
# from parser import args
sys.path.append(".")
from segment_anything import sam_model_registry, SamPredictor
sys.path.append("..")
from pgd_attack import pgd
from viewComplete import maineval
import argparse

def box_to_samples(point, box: tuple, mi, x_samples=1, y_samples=1):
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


def main(args):
    # clear previous data
    # delete_all_files_in_directory('/home/ec2-user/Attack-SAM/proposal_figures_v2')
    image = cv2.imread(args.orig_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_checkpoint = args.checkpoint
    model_type = args.sam_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    box_samples = box_to_samples(args.input_point, args.box, args.mi)
    input_label = np.array([1])
    adv_image = pgd(args, predictor=predictor, image=image, input_points=box_samples,
                            input_label=input_label, device=device, ε=args.epsilon, apply_ssa=args.apply_ssa)


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

    ##################

    args.image_path = args.orig_image
    args.adv_image_path = args.savetorch
    if args.sam_checkpoint == "checkpoints/sam_vit_b_01ec64.pth":
        args.model_type = "vit_b"
    elif args.sam_checkpoint == "checkpoints/sam_vit_h_4b8939.pth":
        args.model_type = "vit_h"
    else:
        args.model_type = "vit_l"
    args.device = "cpu"
    image_name = os.path.splitext(os.path.basename(args.image_path))[0]
    adv_image_name = os.path.splitext(os.path.basename(args.adv_image_path))[0]
    args.pdf_filename = f"PDF_Result/{args.sam_model}_{args.model_type}_{args.apply_ssa}_{args.epsilon}_{args.rho}_{args.mi}/{image_name}_{args.adv}_{args.model_type}_{adv_image_name}.pdf"
    if not os.path.exists(os.path.dirname(args.pdf_filename)):
        os.makedirs(os.path.dirname(args.pdf_filename))
    # args.result_csv = 'SSAscripts/result_2.csv'
    print("Evaluating......")
    maineval(args)


    #######
    #python SSAscripts/attack.py

