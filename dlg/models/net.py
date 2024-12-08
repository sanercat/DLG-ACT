import torch
import torch.nn as nn
from collections import OrderedDict

from .variational_bottleneck import VariationalBottleneck

class CustomActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 前向传播使用 Sigmoid 函数
        ctx.save_for_backward(x)  # 保存 x 以供反向传播使用
        return torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # 反向传播使用 ArcTan 函数的梯度（ArcTan 的导数是 1 / (1 + x^2)）
        grad_input = grad_output / (1 + x ** 2)
        return grad_input
class CustomActivation(nn.Module):
    def __init__(self):
        super(CustomActivation, self).__init__()

    def forward(self, x):
        return CustomActivationFunction.apply(x)

class HybridActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 前向传播使用 ReLU 函数
        ctx.save_for_backward(x)  # 保存 x 以供反向传播使用
        return torch.relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # 反向传播使用 Swish 函数的梯度
        sigmoid_x = torch.sigmoid(x)
        grad_input = grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))
        return grad_input

class HybridActivationLayer(nn.Module):
    def __init__(self):
        super(HybridActivationLayer, self).__init__()

    def forward(self, x):
        return HybridActivationFunction.apply(x)

class LeNet_MNIST(nn.Module):
    """LeNet variant from https://github.com/mit-han-lab/dlg/blob/master/models/vision.py."""
    def __init__(self):
        """3-Layer sigmoid Conv with large linear layer."""
        super(LeNet_MNIST, self).__init__()
        # act = nn.Sigmoid
        # act=CustomActivation
        act = SigmoidWithCustomGradientLayer
        # act = nn.ReLU
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(588, 10)
        )

        self.apply(self.weights_init)

    def forward(self, x):
        out = self.features(x)
        feature = out.view(out.size(0), 588)
        out = self.classifier(feature)
        return out, feature, x

    @staticmethod
    def weights_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # act = nn.Sigmoid
        act=CustomActivation
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(768, 100)
        )
        self.apply(self.weights_init)
        
    def forward(self, x):
        out = self.body(x)
        feature = out.view(out.size(0), -1)
        # print(out.size())
        out = self.fc(feature)
        return out,feature,x
    
    @staticmethod
    def weights_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
class LeNet_CIFAR10(nn.Module):
    def __init__(self):
        super(LeNet_CIFAR10, self).__init__()
        act = nn.Sigmoid  
        # act = CustomActivation  
        

        # 构建卷积层
        self.body = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, padding=5//2, stride=2),  
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )

        # 构建全连接层
        self.fc = nn.Sequential(
            nn.Linear(768, 10)  
        )

        # 初始化权重
        self.apply(self.weights_init)

    def forward(self, x):
        out = self.body(x)
        feature = out.view(out.size(0), -1)  
        out = self.fc(feature)  
        return out, feature, x

    @staticmethod
    def weights_init(m):
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.uniform_(-0.5, 0.5)
# for imprintattack
class LeNet_MNIST_imp(nn.Module):
    def __init__(self):
        """3-Layer sigmoid Conv with large linear layer."""
        super(LeNet_MNIST_imp, self).__init__()
        act = nn.Sigmoid
        # act = nn.ReLU
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(588, 10)
        )

        self.apply(self.weights_init)

    def forward(self, x):
        x_in = x[0].view(x[0].size(0), 1, 28, 28)
        out = self.features(x_in)
        feature = out.view(out.size(0), 588)
        out = self.classifier(feature)
        return out, feature, x[1]

    @staticmethod
    def weights_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)


class LeNet_PRECODE(nn.Module):
    """LeNet variant from https://github.com/mit-han-lab/dlg/blob/master/models/vision.py."""
    def __init__(self, hidden_size, beta=1e-3):
        """3-Layer sigmoid Conv with large linear layer."""
        super(LeNet_PRECODE, self).__init__()
        act = nn.Sigmoid
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
            nn.Flatten(),
        )
        # self.hidden2mu = nn.Linear(588, hidden_size)
        # self.hidden2log_var = nn.Linear(588, hidden_size)
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_size, 588),
        #     nn.Linear(588, 10),
        # )
        # self.hidden_size = hidden_size
        # self.setup = setup

        self.VB = VariationalBottleneck((588,), K=hidden_size, beta=beta)
        self.classifier = nn.Sequential(
            nn.Linear(588, 10)
        )

        self.apply(self.weights_init)

    # def set_z(self, z_value):
    #     self.z = z_value

    # def reparametrize(self, mu, log_var, z):
    #     sigma = torch.exp(0.5*log_var)
    #     self.sigma = sigma
    #     return mu + sigma*z

    def forward(self, x):
        hidden = self.features(x)
        # self.hidden = hidden
        # mu = self.hidden2mu(hidden)
        # log_var = self.hidden2log_var(hidden)
        # self.mu = mu
        # self.log_var = log_var
        # z = torch.randn([x.size()[0],self.hidden_size]).to(**self.setup)
        # feature = self.reparametrize(mu, log_var, z)
        feature = self.VB(hidden)
        out = self.classifier(feature)
        return out, hidden, x

    # calculates the VBLoss
    def loss(self):
        return self.VB.loss()

    @staticmethod
    def weights_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)


# for imprintattack
class LeNet_PRECODE_imp(nn.Module):
    def __init__(self, hidden_size, beta=1e-3):
        """3-Layer sigmoid Conv with large linear layer."""
        super(LeNet_PRECODE_imp, self).__init__()
        act = nn.Sigmoid
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
            nn.Flatten(),
        )
        # self.hidden2mu = nn.Linear(588, hidden_size)
        # self.hidden2log_var = nn.Linear(588, hidden_size)
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_size, 588),
        #     nn.Linear(588, 10),
        # )
        # self.hidden_size = hidden_size
        # self.setup = setup

        self.VB = VariationalBottleneck((588,), K=hidden_size, beta=beta)
        self.classifier = nn.Sequential(
            nn.Linear(588, 10)
        )

        self.apply(self.weights_init)

    # def set_z(self, z_value):
    #     self.z = z_value

    # def reparametrize(self, mu, log_var, z):
    #     sigma = torch.exp(0.5*log_var)
    #     self.sigma = sigma
    #     return mu + sigma*z

    def forward(self, x):
        x_in = x[0].view(x[0].size(0), 1, 28, 28)
        hidden = self.features(x_in)
        # self.hidden = hidden
        # mu = self.hidden2mu(hidden)
        # log_var = self.hidden2log_var(hidden)
        # self.mu = mu
        # self.log_var = log_var
        # z = torch.randn([x_in.size()[0],self.hidden_size]).to(**self.setup)
        # feature = self.reparametrize(mu, log_var, z)
        feature = self.VB(hidden)
        out = self.classifier(feature)
        return out, hidden, x[1]

    # calculates the VBLoss
    def loss(self):
        return self.VB.loss()

    @staticmethod
    def weights_init(m):
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)


class ConvNet(nn.Module):
    """ConvNetBN with custom weight initialization."""

    def __init__(self, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super(ConvNet, self).__init__()
        # act = CustomInPlaceActivation 
        act = nn.ReLU
        # act = HybridActivationLayer
        self.model = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
            ('bn0', nn.BatchNorm2d(1 * width)),
            ('relu0', act()),

            ('conv1', nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn1', nn.BatchNorm2d(2 * width)),
            ('relu1', act()),

            ('conv2', nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm2d(2 * width)),
            ('relu2', act()),

            ('conv3', nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn3', nn.BatchNorm2d(4 * width)),
            ('relu3', act()),

            ('conv4', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn4', nn.BatchNorm2d(4 * width)),
            ('relu4', act()),

            ('conv5', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn5', nn.BatchNorm2d(4 * width)),
            ('relu5', act()),

            ('pool0', nn.MaxPool2d(3)),

            ('conv6', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', nn.BatchNorm2d(4 * width)),
            ('relu6', act()),

            ('conv7', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn7', nn.BatchNorm2d(4 * width)),
            ('relu7', act()),

            ('pool1', nn.MaxPool2d(3))
        ]))
        
        self.fc = nn.Linear(36 * width, num_classes)
        self.act1 = CustomActivation()
        self.act = nn.ReLU()
        
        # 调用权重初始化
        # self.apply(self.weights_init)

    @staticmethod
    def weights_init(m):
        """Custom weight initialization."""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -0.5, 0.5)
            if m.bias is not None:
                nn.init.uniform_(m.bias, -0.5, 0.5)

    def forward(self, input):
        out = self.model(input)
        feature = out.view(out.size(0), -1)
        out = self.fc(feature)
        # out = self.act(out)
        return out, feature, input

class TinyImageNetConvNet(nn.Module):
    """ConvNet for TinyImageNet-200 with custom weight initialization."""

    def __init__(self, width=64, num_classes=200, num_channels=3):
        """Init with width and num classes."""
        super(TinyImageNetConvNet, self).__init__()
        act = CustomActivation  # 假设自定义激活函数已经定义
        self.model = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
            ('bn0', nn.BatchNorm2d(1 * width)),
            ('relu0', nn.ReLU()),

            ('conv1', nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn1', nn.BatchNorm2d(2 * width)),
            ('relu1', nn.ReLU()),

            ('conv2', nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm2d(2 * width)),
            ('relu2', nn.ReLU()),

            ('conv3', nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn3', nn.BatchNorm2d(4 * width)),
            ('relu3', nn.ReLU()),

            ('conv4', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn4', nn.BatchNorm2d(4 * width)),
            ('relu4', nn.ReLU()),

            ('conv5', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn5', nn.BatchNorm2d(4 * width)),
            ('relu5', nn.ReLU()),

            ('pool0', nn.MaxPool2d(3)),  # Downsampling

            ('conv6', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', nn.BatchNorm2d(4 * width)),
            ('relu6', nn.ReLU()),

            ('conv7', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn7', nn.BatchNorm2d(4 * width)),
            ('relu7', act()),

            ('pool1', nn.MaxPool2d(3))  # Further downsampling
        ]))
        
        # TinyImageNet-200 的图片经过上述模型后，尺寸缩小到 (4 * width, 5, 5)
        self.fc = nn.Linear(73728, num_classes)
        self.act1 = CustomActivation()
        
        # 权重初始化
        # self.apply(self.weights_init)

    @staticmethod
    def weights_init(m):
        """Custom weight initialization."""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -0.5, 0.5)
            if m.bias is not None:
                nn.init.uniform_(m.bias, -0.5, 0.5)

    def forward(self, input):
        out = self.model(input)
        feature = out.view(out.size(0), -1)
        out = self.fc(feature)
        # out = self.act1(out)  # 使用自定义激活函数
        return out, feature, input


class ConvNet_imp(nn.Module):
    """ConvNetBN."""

    def __init__(self, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super(ConvNet_imp, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
            ('bn0', nn.BatchNorm2d(1 * width)),
            ('relu0', nn.ReLU()),

            ('conv1', nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn1', nn.BatchNorm2d(2 * width)),
            ('relu1', nn.ReLU()),

            ('conv2', nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm2d(2 * width)),
            ('relu2', nn.ReLU()),

            ('conv3', nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn3', nn.BatchNorm2d(4 * width)),
            ('relu3', nn.ReLU()),

            ('conv4', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn4', nn.BatchNorm2d(4 * width)),
            ('relu4', nn.ReLU()),

            ('conv5', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn5', nn.BatchNorm2d(4 * width)),
            ('relu5', nn.ReLU()),

            ('pool0', nn.MaxPool2d(3)),

            ('conv6', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', nn.BatchNorm2d(4 * width)),
            ('relu6', nn.ReLU()),

            ('conv6', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', nn.BatchNorm2d(4 * width)),
            ('relu6', nn.ReLU()),

            ('conv7', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn7', nn.BatchNorm2d(4 * width)),
            ('relu7', nn.ReLU()),

            ('pool1', nn.MaxPool2d(3))
        ]))
        self.linear = nn.Linear(36 * width, num_classes)

    def forward(self, x):
        x_in = x[0].view(x[0].size(0), 3, 32, 32)
        out = self.model(x_in)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        return out, feature, x[1]


class ConvNet_PRECODE(nn.Module):
    """ConvNetBN."""

    def __init__(self, hidden_size, beta=1e-3, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super(ConvNet_PRECODE, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
            ('bn0', nn.BatchNorm2d(1 * width)),
            ('relu0', nn.ReLU()),

            ('conv1', nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn1', nn.BatchNorm2d(2 * width)),
            ('relu1', nn.ReLU()),

            ('conv2', nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm2d(2 * width)),
            ('relu2', nn.ReLU()),

            ('conv3', nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn3', nn.BatchNorm2d(4 * width)),
            ('relu3', nn.ReLU()),

            ('conv4', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn4', nn.BatchNorm2d(4 * width)),
            ('relu4', nn.ReLU()),

            ('conv5', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn5', nn.BatchNorm2d(4 * width)),
            ('relu5', nn.ReLU()),

            ('pool0', nn.MaxPool2d(3)),

            ('conv6', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', nn.BatchNorm2d(4 * width)),
            ('relu6', nn.ReLU()),

            ('conv6', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', nn.BatchNorm2d(4 * width)),
            ('relu6', nn.ReLU()),

            ('conv7', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn7', nn.BatchNorm2d(4 * width)),
            ('relu7', nn.ReLU()),

            ('pool1', nn.MaxPool2d(3))
        ]))

        # self.hidden2mu = nn.Linear(36 * width, hidden_size)
        # self.hidden2log_var = nn.Linear(36 * width, hidden_size)
        # self.hidden_size = hidden_size
        # self.setup = setup
        # self.linear = nn.Sequential(
        #     nn.Linear(hidden_size, 36 * width),
        #     nn.Linear(36 * width, num_classes),
        # )

        self.VB = VariationalBottleneck((36 * width,), K=hidden_size, beta=beta)
        self.linear = nn.Linear(36 * width, num_classes)

    # def reparametrize(self, mu, log_var, z):
    #     sigma = torch.exp(0.5*log_var)
    #     self.sigma = sigma
    #     return mu + sigma*z

    def forward(self, input):
        out = self.model(input)
        hidden = out.view(out.size(0), -1)
        # precode
        # self.hidden = hidden
        # mu = self.hidden2mu(hidden)
        # log_var = self.hidden2log_var(hidden)
        # self.mu = mu
        # self.log_var = log_var
        # z = torch.randn([input.size()[0], self.hidden_size]).to(**self.setup)
        # feature = self.reparametrize(mu, log_var, z)
        feature = self.VB(hidden)
        # classify
        out = self.linear(feature)
        return out, feature, input

    # calculates the VBLoss
    def loss(self):
        return self.VB.loss()


class ConvNet_PRECODE_imp(nn.Module):
    """ConvNetBN."""

    def __init__(self, hidden_size, beta=1e-3, width=32, num_classes=10, num_channels=3):
        """Init with width and num classes."""
        super(ConvNet_PRECODE_imp, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_channels, 1 * width, kernel_size=3, padding=1)),
            ('bn0', nn.BatchNorm2d(1 * width)),
            ('relu0', nn.ReLU()),

            ('conv1', nn.Conv2d(1 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn1', nn.BatchNorm2d(2 * width)),
            ('relu1', nn.ReLU()),

            ('conv2', nn.Conv2d(2 * width, 2 * width, kernel_size=3, padding=1)),
            ('bn2', nn.BatchNorm2d(2 * width)),
            ('relu2', nn.ReLU()),

            ('conv3', nn.Conv2d(2 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn3', nn.BatchNorm2d(4 * width)),
            ('relu3', nn.ReLU()),

            ('conv4', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn4', nn.BatchNorm2d(4 * width)),
            ('relu4', nn.ReLU()),

            ('conv5', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn5', nn.BatchNorm2d(4 * width)),
            ('relu5', nn.ReLU()),

            ('pool0', nn.MaxPool2d(3)),

            ('conv6', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', nn.BatchNorm2d(4 * width)),
            ('relu6', nn.ReLU()),

            ('conv6', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn6', nn.BatchNorm2d(4 * width)),
            ('relu6', nn.ReLU()),

            ('conv7', nn.Conv2d(4 * width, 4 * width, kernel_size=3, padding=1)),
            ('bn7', nn.BatchNorm2d(4 * width)),
            ('relu7', nn.ReLU()),

            ('pool1', nn.MaxPool2d(3))
        ]))

        # self.hidden2mu = nn.Linear(36 * width, hidden_size)
        # self.hidden2log_var = nn.Linear(36 * width, hidden_size)
        # self.hidden_size = hidden_size
        # self.setup = setup
        # self.linear = nn.Sequential(
        #     nn.Linear(hidden_size, 36 * width),
        #     nn.Linear(36 * width, num_classes),
        # )

        self.VB = VariationalBottleneck((36 * width,), K=hidden_size, beta=beta)
        self.linear = nn.Linear(36 * width, num_classes)

    # def reparametrize(self, mu, log_var, z):
    #     sigma = torch.exp(0.5*log_var)
    #     self.sigma = sigma
    #     return mu + sigma*z

    def forward(self, x):
        x_in = x[0].view(x[0].size(0), 3, 32, 32)
        out = self.model(x_in)
        hidden = out.view(out.size(0), -1)
        # precode
        # self.hidden = hidden
        # mu = self.hidden2mu(hidden)
        # log_var = self.hidden2log_var(hidden)
        # self.mu = mu
        # self.log_var = log_var
        # z = torch.randn([x_in.size()[0], self.hidden_size]).to(**self.setup)
        # feature = self.reparametrize(mu, log_var, z)
        feature = self.VB(hidden)
        # classify
        out = self.linear(feature)
        return out, feature, x[1]

    # calculates the VBLoss
    def loss(self):
        return self.VB.loss()

class VGGNet16(nn.Module):
    def __init__(self,num_classes=10):
        super(VGGNet16, self).__init__()

        self.Conv1 = nn.Sequential(
            # CIFAR10 数据集是彩色图 - RGB三通道, 所以输入通道为 3, 图片大小为 32*32
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            # inplace-选择是否对上层传下来的tensor进行覆盖运算, 可以有效地节省内存/显存
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 池化层
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.Conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 全连接层
        self.fc = nn.Sequential(

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            # 使一半的神经元不起作用，防止参数量过大导致过拟合
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(128, num_classes)
        )

    def forward(self, input):
        # 五个卷积层
        x = self.Conv1(input)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Conv5(x)

        # 数据平坦化处理，为接下来的全连接层做准备
        feature = x.view(-1, 512)
        out = self.fc(feature)
        return out,feature,input
