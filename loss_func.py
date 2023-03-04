import numpy as np
import torch
import torch.nn as nn


def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)

    return entropy 


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None, simple_con=False, device='cpu'):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        if simple_con:
            op_out = torch.cat((softmax_output, feature), 1)
            ad_out = ad_net(op_out, coeff)
        else:
            op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
            ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)), coeff)
    else:
        pass
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(device)
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCEWithLogitsLoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCEWithLogitsLoss()(ad_out, dc_target)
