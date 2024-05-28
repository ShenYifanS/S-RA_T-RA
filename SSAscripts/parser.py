import argparse

parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--orig_image', type=str, default='notebooks/images/truck.jpg',
                    help='[truck.jpg]')
parser.add_argument('--sam_model', type=str, default='vit_b',
                    help='[vit_b, vit_h]')
parser.add_argument('--checkpoint', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                    help='[checkpoints/sam_vit_b_01ec64.pth, sam_vit_h_4b8939.pth]')
parser.add_argument('--box', type=tuple, default=(400, 500, 350, 400),
                    help='(minx, maxx, miny, maxy)')
parser.add_argument('--epsilon', type=float, default=8.,
                    help='2, 4, 8, 16')
parser.add_argument('--apply_ssa', type=bool, default=False,
                    help='True or False')
args = parser.parse_args()