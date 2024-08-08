import matplotlib.pyplot as plt
from quick_test.plot_utils import *
import os
import torch
def run_test(input_point, input_label, predictor, image_np, out_dir, device, phase='after', dpi=100):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    transformed_image = predictor.transform.apply_image(image_np)# X, 1024, 3

    # hwc transform
    torch_image = torch.from_numpy(transformed_image).permute(2, 0, 1).contiguous()[None].to(device)
    predictor.set_torch_image(torch_image, image_np.shape[:2])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
        # return_logits=True,
    )
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image_np)
        # plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        # plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        if phase=='after':
            plt.savefig(os.path.join(out_dir, f'output_{i}_after.png'), dpi = dpi, bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig(os.path.join(out_dir, f'output_{i}.png'), dpi = dpi, bbox_inches='tight', pad_inches=0)
        break
