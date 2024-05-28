import os

import torch
import sys
sys.path.append("..")

from torch import Tensor, nn
from torch.autograd import grad, Variable
from torch.nn import functional as F

# from parser import args
from ssa import image_transfer


def pgd(args, predictor, image, input_points, input_label, device, ε, apply_ssa=False,
        restarts=1, neg_th=-10, num_steps=100, relative_step_size=1/100):
    ε = torch.full((1,), ε, dtype=torch.float, device=device)

    adv_image = None
    best_loss = float('inf')

    for i in range(restarts):
        adv_loss_run, adv_image_run = _pgd(args, predictor=predictor, image=image, input_points=input_points,
                      input_label=input_label, device=device, ε=ε, apply_ssa=apply_ssa, neg_th=neg_th,
                      num_steps=num_steps, relative_step_size=relative_step_size, random_init=(i!=0))
        if adv_loss_run <= best_loss:
            best_loss = adv_loss_run
            adv_image = adv_image_run

    return adv_image


def _pgd(args, predictor, image, input_points, input_label, device, ε, apply_ssa=False,
         neg_th=-10, num_steps=100, relative_step_size=1/100, random_init=False):
    loss_function = nn.MSELoss()
    input_image = predictor.transform.apply_image(image)
    # 684, 1024, 3, 255 scale
    torch_image = torch.as_tensor(input_image, dtype=torch.float32, device=device)
    ssa_image = torch_image

    # 255 scale
    batch_view = lambda tensor: tensor.view(1, *[1] * (torch_image.ndim - 1))
    neg_inputs = -torch_image
    one_minus_inputs = 255. - torch_image

    step_size = ε * relative_step_size

    δ = torch.zeros_like(torch_image, requires_grad=True)
    best_loss = float('inf')
    best_adv = torch_image.clone()

    if random_init:
        δ.data.uniform_().sub_(0.5).mul_(2 * batch_view(ε))
    N = 1
    if apply_ssa:
        print("SSA......")
        N = 5
    for i in range(num_steps):
        # if apply_ssa:
        #     ssa_image = image_transfer(predictor, input_points, input_label, torch_image, ε, device=device)
        delta = torch.zeros_like(δ)
        for n in range(N):
            if apply_ssa:
                ssa_image = image_transfer(predictor, input_points, input_label, torch_image, ε, device=device, rho = args.rho)
            adv_image = ssa_image + δ
            adv_image = adv_image.permute(2, 0, 1).contiguous()[None, :, :, :]
            predictor.set_torch_image(adv_image, image.shape[:2])
            # total_loss = 0
            # for input_point in input_points:
            #     masks, scores, logits = predictor.predict(
            #         point_coords=input_point,
            #         point_labels=input_label,
            #         multimask_output=True,
            #         return_logits=True
            #     )
            #     neg_threshold = torch.full(masks.size(), neg_th, dtype=torch.float32).to(device=device)
            #     loss = loss_function(torch.clamp(masks, min=neg_th, max=None), neg_threshold).to(device=device)
            #     total_loss += loss
            # adv_loss = total_loss / input_points.shape[0]
            # δ_grad = - grad(adv_loss.sum(), δ, only_inputs=True)[0]
            # δ.data.add_(batch_view(step_size) * δ_grad.sign()).clamp_(min=batch_view(-ε), max=batch_view(ε))
            # δ.data.clamp_(min=neg_inputs, max=one_minus_inputs)
            batch_size = 1  # 每批处理的点数
            num_points = input_points.shape[0]
            num_batches = (num_points + batch_size - 1) // batch_size  # 计算总批次数
            accumulated_loss = 0.0
            for j in range(num_batches):
                start_idx = j * batch_size
                end_idx = min((j + 1) * batch_size, num_points)
                batch_points = input_points[start_idx:end_idx]
                total_loss = 0.0
                for input_point in batch_points:
                    masks, scores, logits = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=True,
                        return_logits=True
                    )
                    neg_threshold = torch.full(masks.size(), neg_th, dtype=torch.float32).to(device=device)
                    loss = loss_function(torch.clamp(masks, min=neg_th, max=None), neg_threshold).to(device=device)
                    total_loss += loss
                    del masks, scores, logits, neg_threshold, loss
                # batch_average_loss = total_loss / len(batch_points)
                # print(total_loss)
                # print(f'Batch {j} average loss: {batch_average_loss}')
                accumulated_loss += total_loss
                torch.cuda.empty_cache()
            adv_loss = accumulated_loss / num_points
            δ_grad = -torch.autograd.grad(adv_loss, δ, only_inputs=True, retain_graph=False)[0]
            δ.data.add_(batch_view(step_size) * δ_grad.sign()).clamp_(min=batch_view(-ε), max=batch_view(ε))
            δ.data.clamp_(min=neg_inputs, max=one_minus_inputs)

            delta = delta + δ
        δ = delta / N

        # batch_size = 1  # 每批处理的点数
        # num_points = input_points.shape[0]
        # num_batches = (num_points + batch_size - 1) // batch_size  # 计算总批次数
        # accumulated_loss = 0.0
        # for j in range(num_batches):
        #     start_idx = j * batch_size
        #     end_idx = min((j + 1) * batch_size, num_points)
        #     batch_points = input_points[start_idx:end_idx]
        #     total_loss = 0.0
        #     for input_point in batch_points:
        #         masks, scores, logits = predictor.predict(
        #             point_coords=input_point,
        #             point_labels=input_label,
        #             multimask_output=True,
        #             return_logits=True
        #         )
        #         neg_threshold = torch.full(masks.size(), neg_th, dtype=torch.float32).to(device=device)
        #         loss = loss_function(torch.clamp(masks, min=neg_th, max=None), neg_threshold).to(device=device)
        #         total_loss += loss
        #         del masks, scores, logits, neg_threshold, loss
        #     # batch_average_loss = total_loss / len(batch_points)
        #     # print(total_loss)
        #     # print(f'Batch {j} average loss: {batch_average_loss}')
        #     accumulated_loss += total_loss
        #     torch.cuda.empty_cache()
        # adv_loss = accumulated_loss / num_points
        # δ_grad = -torch.autograd.grad(adv_loss, δ, only_inputs=True, retain_graph=False)[0]
        # δ.data.add_(batch_view(step_size) * δ_grad.sign()).clamp_(min=batch_view(-ε), max=batch_view(ε))
        # δ.data.clamp_(min=neg_inputs, max=one_minus_inputs)

        print(f'loss at step {i}: {adv_loss}')
        if adv_loss <= best_loss:
            best_loss = adv_loss
            best_adv = torch_image + δ
            # torch.save(torch_image + δ, 'adv_image_{0}_epsilon_{1}.pt'.format(args.sam_model, args.epsilon))

            # 684

            # TODO: it doesn't make sense, should clamp
            # TODO: the delta should be applied to torch_image
            best_adv = best_adv.unsqueeze(0)
            best_adv = best_adv.permute(0, 3, 1, 2)
            adv_image = torch.clamp(best_adv, 0, 255)

            # print(adv_image.size())

            if not os.path.exists(os.path.dirname(args.savetorch)):
                os.makedirs(os.path.dirname(args.savetorch))
            torch.save(adv_image, args.savetorch)
        # print(step_size.size(), ε.size())


    return best_loss, best_adv
