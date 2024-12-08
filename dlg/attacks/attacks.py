import torch
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# for imprint attack
from utils.util import (
    attack_cfg_default,
    mnist_data_cfg_default,
    cifar10_data_cfg_default,
    skin_data_cfg_default,
    imagenet_data_cfg_default,
    celeba32_data_cfg_default,
    tinyimagenet_data_cfg_default
    )

from .base.imprint import ImprintBlock, SparseImprintBlock
from .base.analytic_attack import ImprintAttacker


def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

# compute tv
def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy

# DLG and GS attack
def gradient_closure(args, model, optimizer, loss_fn, dummy_data, label, origin_gradient):
    def closure():
        # optimizer.zero_grad()
        # model.zero_grad()

        # out, _, _ = model(dummy_data)
        # loss = loss_fn(out, label)
        # dummy_gradient = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        # rec_loss = 0
        # pnorm = [0, 0]
        if args.attack == 'dlg':
            optimizer.zero_grad()
            model.zero_grad()
            out, _, _ = model(dummy_data)
            dummy_onehot_label = F.softmax(label, dim=-1)
            loss=cross_entropy_for_onehot(out,dummy_onehot_label)
            # loss = loss_fn(out, label)
            dummy_gradient = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            rec_loss = 0
            pnorm = [0, 0]
            for gx, gy in zip(dummy_gradient, origin_gradient):
                rec_loss += ((gx - gy).pow(2)).sum()
        elif args.attack == 'gs':
            optimizer.zero_grad()
            model.zero_grad()

            out, _, _ = model(dummy_data)
            loss = loss_fn(out, label)
            dummy_gradient = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            rec_loss = 0
            pnorm = [0, 0]
            for gx, gy in zip(dummy_gradient, origin_gradient):
                rec_loss -= (gx * gy).sum()  
                pnorm[0] += gx.pow(2).sum()
                pnorm[1] += gy.pow(2).sum()
            rec_loss = 1 + rec_loss / torch.sqrt(pnorm[0]) / torch.sqrt(pnorm[1])

        if args.tv > 0:
            rec_loss += args.tv * total_variation(dummy_data)  
        rec_loss.backward()

        return rec_loss

    return closure


#rnn
# def gradient_closure(args, model, optimizer, loss_fn, dummy_data, label, origin_gradient):
#     def closure():
#         optimizer.zero_grad()
#         model.zero_grad()

#         # 在前向传播过程中禁用 CuDNN
#         with torch.backends.cudnn.flags(enabled=False):
#             out, _, _ = model(dummy_data)  # 前向传播
#             loss = loss_fn(out, label)  # 计算损失

#         # 计算模型的梯度，使用 create_graph=True 以便支持二阶导数
#         dummy_gradient = torch.autograd.grad(loss, model.parameters(), create_graph=True)

#         rec_loss = 0
#         pnorm = [0, 0]

#         # DLG 攻击
#         if args.attack == 'dlg':
#             for gx, gy in zip(dummy_gradient, origin_gradient):
#                 rec_loss += ((gx - gy).pow(2)).sum()

#         # GS 攻击
#         elif args.attack == 'gs':
#             for gx, gy in zip(dummy_gradient, origin_gradient):
#                 rec_loss -= (gx * gy).sum()
#                 pnorm[0] += gx.pow(2).sum()
#                 pnorm[1] += gy.pow(2).sum()
#             # 避免除以零的情况
#             rec_loss = 1 + rec_loss / torch.sqrt(pnorm[0]) / torch.sqrt(pnorm[1])

#         # 添加 TV 正则化项
#         if args.tv > 0:
#             rec_loss += args.tv * total_variation(dummy_data)

#         rec_loss.backward()  # 反向传播

#         return rec_loss

#     return closure


#SNN
# def gradient_closure(args, model, optimizer, loss_fn, dummy_data, label, origin_gradient):
#     def closure():
#         functional.reset_net(model)  # 重置网络状态，适用于 SNN 等需要重置状态的网络
#         optimizer.zero_grad()
#         model.zero_grad()

#         # 计算模型的输出和损失
#         out, _, _ = model(dummy_data)
#         loss = loss_fn(out, label)

#         # 计算模型的梯度
#         dummy_gradient = torch.autograd.grad(loss, model.parameters(), allow_unused=True)

#         rec_loss = 0
#         pnorm = [0, 0]

#         # DLG 攻击
#         if args.attack == 'dlg':
#             for gx, gy in zip(dummy_gradient, origin_gradient):
#                 if gx is not None and gy is not None:  # 检查 gx 和 gy 是否为 None
#                     rec_loss += ((gx - gy).pow(2)).sum()
        
#         # GS 攻击
#         elif args.attack == 'gs':
#             for gx, gy in zip(dummy_gradient, origin_gradient):
#                 if gx is not None and gy is not None:  # 检查 gx 和 gy 是否为 None
#                     rec_loss -= (gx * gy).sum()
#                     pnorm[0] += gx.pow(2).sum()
#                     pnorm[1] += gy.pow(2).sum()
#             # 避免除以零的情况
#             if pnorm[0] > 0 and pnorm[1] > 0:
#                 rec_loss = 1 + rec_loss / torch.sqrt(pnorm[0]) / torch.sqrt(pnorm[1])

#         # 添加 TV 正则化项（如果设置了 args.tv）
#         if args.tv > 0:
#             rec_loss += args.tv * total_variation(dummy_data)
        
#         rec_loss.backward()  # 计算反向传播

#         return rec_loss

#     return closure

def DLG_attack(args, gt_gradient, gt_images, gt_labels, model, loss_fn, dm, ds, device):

    # initialize
    if args.prior > -1: # prior knowledge (average image)
        print('Advanced attack, using prior knowledge')
        if args.dataset == 'CIFAR10':
            data_mean = (0.4914, 0.4822, 0.4465)
            data_std = (0.247, 0.243, 0.261)
            data_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(data_mean, data_std)
                                ])
            img = Image.open('./attacks/AvgImgs/cifar10_y{}.png'.format(args.prior)).convert('RGB')
            dummy_data = data_transform(img).expand(gt_images.size()[0], 3, 32, 32).to(device).requires_grad_(True)
        elif args.dataset == 'MNIST':
            data_mean = (0.13066047430038452, )
            data_std = (0.30810782313346863,)
            data_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(data_mean, data_std)
                                ])
            img = Image.open('./attacks/AvgImgs/mnist_y{}.png'.format(args.prior)).convert('L')
            dummy_data = data_transform(img).expand(gt_images.size()[0], 1, 28, 28).to(device).requires_grad_(True)
        else:
            assert False, 'No prior knowledge for this dataset.'
    elif args.attack == 'dlg':
        dummy_data = torch.randn(gt_images.size()).to(device).requires_grad_(True)
        # print(gt_labels)
        gt_label = gt_labels.view(1, ).long().to(device)
        gt_onehot_label = label_to_onehot(gt_label).to(device)
        print(gt_onehot_label.size())
        gt_labels = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
    
        # 尝试重新定义 gt_labels
        # gt_labels = torch.randn(gt_onehot_label.size(), device=device, requires_grad=True)   
    else:
        dummy_data = torch.randn(gt_images.size()).to(device).requires_grad_(True)

    if args.attack == 'dlg':
        optimizer = torch.optim.LBFGS([dummy_data,gt_labels])
    else:
        optimizer = torch.optim.Adam([dummy_data], lr=args.lr)

    if args.lr_decay:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[args.max_iter // 2.667,
                                                                     args.max_iter // 1.6,
                                                                     args.max_iter // 1.142],
                                                         gamma=0.1)   # 3/8 5/8 7/8
    # reconstruct
    for iter in range(args.max_iter):
        closure = gradient_closure(args, model, optimizer, loss_fn,
                                   dummy_data, gt_labels, gt_gradient)
        rec_loss = optimizer.step(closure)
        if args.lr_decay:
            scheduler.step()
        # with torch.no_grad():
        #     # Project into image space
        #     if args.boxed:
        #         dummy_data = torch.clamp(dummy_data, -dm / ds, (1 - dm) / ds)
        if (iter + 1 == args.max_iter) or iter % 10 == 0:
            print(f'Attack - Iter-{iter}: Rec_loss-{rec_loss.item():2.4f}.')

    return dummy_data.detach()


# ImprintAttack and SparseImprintAttack
def Imprint_setting(args, model, loss_fn, setup):
    if args.dataset == 'MNIST':
        data_cfg_default = mnist_data_cfg_default
    elif args.dataset == 'CIFAR10':
        data_cfg_default = cifar10_data_cfg_default
    elif args.dataset == 'HAM10000':
        data_cfg_default = skin_data_cfg_default
    elif args.dataset == 'ImageNet':
        data_cfg_default = imagenet_data_cfg_default
    elif args.dataset == 'CelebA':
        data_cfg_default = celeba32_data_cfg_default
    elif args.dataset == 'TinyImageNet':
        data_cfg_default = tinyimagenet_data_cfg_default

    # Load imprint module
    input_dim = data_cfg_default.shape[0] * data_cfg_default.shape[1] * data_cfg_default.shape[2]
    num_bins = args.bins # Here we define number of imprint bins
    if args.imprint == 'Sparse':
        block = SparseImprintBlock(input_dim, num_bins=num_bins)
    else:
        block = ImprintBlock(input_dim, num_bins=num_bins)

    # Modified model
    model = torch.nn.Sequential(
        torch.nn.Flatten(), block, model
    )
    model.to(**setup)

    secret = dict(weight_idx=0, bias_idx=1, shape=tuple(data_cfg_default.shape), structure=block.structure)
    secrets = {"ImprintBlock": secret}
    # This is the attacker:
    attacker = ImprintAttacker(model, loss_fn, attack_cfg_default, setup)

    # Server-side computation:
    queries = [dict(parameters=[p for p in model.parameters()], buffers=[b for b in model.buffers()])]
    server_payload = dict(queries=queries, data=data_cfg_default)

    return model, attacker, server_payload, secrets

def Robbing_attack(gt_gradient, gt_labels, attacker, server_payload, secrets):

    shared_data = dict(
        gradients=[gt_gradient],
        buffers=None,
        num_data_points=1,
        labels=gt_labels,
        local_hyperparams=None,
    )
    # Attack:
    reconstructed_user_data, stats = attacker.reconstruct(server_payload, shared_data, secrets, dryrun=False)

    return reconstructed_user_data["data"].detach()


# GGL attack
def gradient_closure_ggl(args, generator, model, optimizer, loss_fn, dummy_data, label, origin_gradient):
    def closure():
        optimizer.zero_grad()
        model.zero_grad()
        generator.zero_grad()

        out, _, _ = model(generator(dummy_data))
        loss = loss_fn(out, label)
        dummy_gradient = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        rec_loss = 0
        pnorm = [0, 0]
        KL=1e-4
        for gx, gy in zip(dummy_gradient, origin_gradient):
            rec_loss -= (gx * gy).sum()
            pnorm[0] += gx.pow(2).sum()
            pnorm[1] += gy.pow(2).sum()
        rec_loss = 1 + rec_loss / torch.sqrt(pnorm[0]) / torch.sqrt(pnorm[1])

        if args.tv > 0:
            rec_loss += args.tv * total_variation(generator(dummy_data))
        # kl loss, use 1e-4 for sim loss and 1e-1 for l2 loss
        KLD = -0.5 * torch.sum(1 + torch.log(torch.std(dummy_data, unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(dummy_data, axis=-1).pow(2) - torch.std(dummy_data, unbiased=False, axis=-1).pow(2))
        rec_loss += KL * KLD
        rec_loss.backward()

        return rec_loss

    return closure

def GGl_attack(args, generator, gt_gradient, gt_images, gt_labels, model, loss_fn, dm, ds, device):

    # initialize
    dummy_data = torch.randn((gt_images.size(0), 128)).to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([dummy_data], lr=args.lr)

    if args.lr_decay:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[args.max_iter // 2.667,
                                                                     args.max_iter // 1.6,
                                                                     args.max_iter // 1.142],
                                                         gamma=0.1)   # 3/8 5/8 7/8
    # reconstruct
    for iter in range(args.max_iter):
        closure = gradient_closure_ggl(args, generator, model, optimizer, loss_fn,
                                       dummy_data, gt_labels, gt_gradient)
        rec_loss = optimizer.step(closure)
        if args.lr_decay:
            scheduler.step()
        # with torch.no_grad():
        #     # Project into image space
        #     if args.boxed:
        #         dummy_data = torch.clamp(dummy_data, -dm / ds, (1 - dm) / ds)
        if (iter + 1 == args.max_iter) or iter % 1000 == 0:
            print(f'Attack - Iter-{iter}: Rec_loss-{rec_loss.item():2.4f}.')

    return dummy_data.detach()
