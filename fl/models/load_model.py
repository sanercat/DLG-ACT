import torch

from .net import LeNet_MNIST, ConvNet, ConvNet_PRECODE, LeNet_PRECODE,LeNet_CIFAR10,CIFAR10ConvNet
from .resnet import resnet34,resnet18
# from .actresnet import resnet18

def load_model(args, setup):

    if args.dataset == 'MNIST':
        model = LeNet_MNIST()
        if args.defense == 'precode':
            model = LeNet_PRECODE(args.precode_size, beta=args.beta)
    elif args.dataset == 'CIFAR10':
        model = ConvNet(width=32, num_classes=10, num_channels=3)
        # model = LeNet_CIFAR10()
        # model = resnet34(pretrained=args.pretrained,num_classes=10)
        # model = CIFAR10ConvNet(width=32, num_classes=10, num_channels=3)
        # model = resnet34(pretrained=args.pretrained,num_classes=10)
        if args.defense == 'precode':
            model = ConvNet_PRECODE(args.precode_size, beta=args.beta, width=32, num_classes=10, num_channels=3)
    elif args.dataset == 'CIFAR100':
        # model = ConvNet(width=32, num_classes=100, num_channels=3)
        # model = LeNet_CIFAR10()
        model = resnet34(pretrained=args.pretrained,num_classes=100)
        # model = CIFAR10ConvNet(width=32, num_classes=100, num_channels=3)
        if args.defense == 'precode':
            model = ConvNet_PRECODE(args.precode_size, beta=args.beta, width=32, num_classes=10, num_channels=3)
    elif args.dataset == 'CelebA':
        model = ConvNet(width=32, num_classes=2, num_channels=3)
    elif args.dataset == 'ImageNet':
        model = resnet18(pretrained=args.pretrained)
    elif args.dataset == 'HAM10000':
        model = resnet18(pretrained=args.pretrained)
        fc = getattr(model, 'fc')
        feature_dim = fc.in_features
        setattr(model,'fc', torch.nn.Linear(feature_dim, 7))
    elif args.dataset == 'TinyImageNet':
        model = resnet18(pretrained=args.pretrained)
        fc = getattr(model, 'fc')
        feature_dim = fc.in_features
        setattr(model,'fc', torch.nn.Linear(feature_dim, 200))

    model.to(**setup)
    return model

