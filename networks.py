import torch
import torch.nn as nn
import torchvision


EPS = 1e-7


class Encoder(nn.Module):
    def __init__(self, cin, cout, nf=64, activation=nn.Tanh):
        super(Encoder, self).__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, cout, kernel_size=1, stride=1, padding=0, bias=False)]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def set_requires_grad(self, grad):
        for param in self.parameters():
            param.requires_grad = grad

    def forward(self, input):
        return self.network(input).reshape(input.size(0),-1)

class EDDeconv(nn.Module):
    def __init__(self, cin, cout, zdim=128, nf=64, activation=nn.Tanh):
        super(EDDeconv, self).__init__()
        ## downsampling
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True)]
        ## upsampling
        network += [
            nn.ConvTranspose2d(zdim, nf*8, kernel_size=4, stride=1, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*8, nf*8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*4, nf*4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf*2, nf*2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, cout, kernel_size=5, stride=1, padding=2, bias=False)]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input)


class ConfNet(nn.Module):
    def __init__(self, cin, cout, zdim=128, nf=64):
        super(ConfNet, self).__init__()
        ## downsampling
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*4, nf*8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf*8, zdim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            nn.ReLU(inplace=True)]
        ## upsampling
        network += [
            nn.ConvTranspose2d(zdim, nf*8, kernel_size=4, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*8, nf*4, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(16*4, nf*4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 16x16
            nn.GroupNorm(16*2, nf*2),
            nn.ReLU(inplace=True)]
        self.network = nn.Sequential(*network)

        out_net1 = [
            nn.ConvTranspose2d(nf*2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 64x64
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, 2, kernel_size=5, stride=1, padding=2, bias=False),  # 64x64
            nn.Softplus()]
        self.out_net1 = nn.Sequential(*out_net1)

        out_net2 = [nn.Conv2d(nf*2, 2, kernel_size=3, stride=1, padding=1, bias=False),  # 16x16
                    nn.Softplus()]
        self.out_net2 = nn.Sequential(*out_net2)

    def set_requires_grad(self, grad):
        for param in self.parameters():
            param.requires_grad = grad

    def forward(self, input):
        out = self.network(input)
        return self.out_net1(out), self.out_net2(out)


class PerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(PerceptualLoss, self).__init__()
        mean_rgb = torch.FloatTensor([0.485, 0.456, 0.406])
        std_rgb = torch.FloatTensor([0.229, 0.224, 0.225])
        self.register_buffer('mean_rgb', mean_rgb)
        self.register_buffer('std_rgb', std_rgb)

        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def normalize(self, x):
        out = x/2 + 0.5
        out = (out - self.mean_rgb.view(1,3,1,1)) / self.std_rgb.view(1,3,1,1)
        return out

    def __call__(self, im1, im2, mask=None, conf_sigma=None):
        im = torch.cat([im1,im2], 0)
        im = self.normalize(im)  # normalize input

        ## compute features
        feats = []
        f = self.slice1(im)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice2(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice3(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice4(f)
        feats += [torch.chunk(f, 2, dim=0)]

        losses = []
        for f1, f2 in feats[2:3]:  # use relu3_3 features only
            loss = (f1-f2)**2
            if conf_sigma is not None:
                loss = loss / (2*conf_sigma**2 +EPS) + (conf_sigma +EPS).log()
            if mask is not None:
                b, c, h, w = loss.shape
                _, _, hm, wm = mask.shape
                sh, sw = hm//h, wm//w
                mask0 = nn.functional.avg_pool2d(mask, kernel_size=(sh,sw), stride=(sh,sw)).expand_as(loss)
                loss = (loss * mask0).sum() / mask0.sum()
            else:
                loss = loss.mean()
            losses += [loss]
        return sum(losses)


class FaceModelGeometry(nn.Module):
    def __init__(self,LoR_K, requires_grad=True):
        super(FaceModelGeometry, self).__init__()
        self.LoR_K = LoR_K
        self.SubGeometry = nn.Parameter(torch.zeros((self.LoR_K, 64, 64)).float())
        self.SubExpression = nn.Parameter(torch.zeros((self.LoR_K, 64, 64)).float())

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, id_code, ex_code, vn):
        subGeo = torch.Tensor.repeat(self.SubGeometry.unsqueeze(0), (vn, 1, 1, 1))
        subExp = torch.Tensor.repeat(self.SubExpression.unsqueeze(0), (vn, 1, 1, 1))
        canon_depth_video = torch.sum(
            id_code.unsqueeze(2).unsqueeze(3) * subGeo + ex_code.unsqueeze(2).unsqueeze(
                3) * subExp, dim=1)
        return canon_depth_video


class FaceModelAlbedo(nn.Module):
    def __init__(self, LoR_K, requires_grad=True):
        super(FaceModelAlbedo, self).__init__()
        self.LoR_K = LoR_K
        self.SubAlbedo = nn.Parameter(torch.zeros((self.LoR_K, 3, 64, 64)).float())

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, a_code, vn):
        subAlbedo = torch.Tensor.repeat(self.SubAlbedo.unsqueeze(0), (vn, 1, 1, 1, 1))
        canon_albed_video = torch.sum(a_code.unsqueeze(2).unsqueeze(3).unsqueeze(4) * subAlbedo, dim=1)
        return canon_albed_video


class FaceModelNet(nn.Module):
    def __init__(self, in_dim, cout, nf=64, activation=nn.Tanh, requires_grad=True):
        super(FaceModelNet, self).__init__()
        prenet = [
            nn.Linear(in_dim, nf),
            nn.ReLU(inplace=True)
        ]
        self.prenet = nn.Sequential(*prenet)

        network = [
            nn.ConvTranspose2d(nf, nf * 8, kernel_size=4, stride=1, padding=0, bias=False),  # 1x1 -> 4x4
            nn.ReLU(inplace=True),
            nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf * 8, nf * 4, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4 -> 8x8
            nn.GroupNorm(16 * 4, nf * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf * 4, nf * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16 * 4, nf * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf * 4, nf * 2, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 16x16
            nn.GroupNorm(16 * 2, nf * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf * 2, nf * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16 * 2, nf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf * 2, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 32x32
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32x32 -> 64x64
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kernel_size=5, stride=1, padding=2, bias=False),
            nn.GroupNorm(16, nf),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, cout, kernel_size=5, stride=1, padding=2, bias=False)
            ]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def set_requires_grad(self, grad):
        for param in self.parameters():
            param.requires_grad = grad

    def forward(self, input):
        x = self.prenet(input).reshape((input.size(0), -1, 1, 1))
        x = self.network(x)
        return x


class TemporalNet(nn.Module):
    def __init__(self, in_dim, out_dim, layer_num):
        super(TemporalNet, self).__init__()
        self.rnn = nn.GRU(in_dim, out_dim, layer_num)

    def load_state_dict(self, state_dict):
        try:
            self.rnn.load_state_dict(state_dict)
        except:
            for k in list(state_dict.keys()):
                nk = k.replace("rnn.", "")
                state_dict[nk] = state_dict.pop(k)

    def set_requires_grad(self, grad):
        for param in self.parameters():
            param.requires_grad = grad

    def forward(self, input):
        return self.rnn(input)


class DecompositionNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DecompositionNet, self).__init__()
        self.FC = nn.Linear(in_dim, out_dim)

    def load_state_dict(self, state_dict):
        try:
            self.FC.load_state_dict(state_dict)
        except:
            for k in list(state_dict.keys()):
                nk = k.replace("FC.", "")
                state_dict[nk] = state_dict.pop(k)

    def set_requires_grad(self, grad):
        for param in self.parameters():
            param.requires_grad = grad

    def forward(self, input):
        return self.FC(input)


class FaceEmbeddingNet(nn.Module):
    def __init__(self, cin, out_dim, nf=64, activation=nn.Tanh, requires_grad=True):
        super(FaceEmbeddingNet, self).__init__()
        network = [
            nn.Conv2d(cin, nf, kernel_size=4, stride=2, padding=1, bias=False),  # 64x64 -> 32x32
            nn.GroupNorm(16, nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf * 2, kernel_size=4, stride=2, padding=1, bias=False),  # 32x32 -> 16x16
            nn.GroupNorm(16 * 2, nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 2, nf * 4, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16 -> 8x8
            nn.GroupNorm(16 * 4, nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 4, nf * 8, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8 -> 4x4
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 8, out_dim, kernel_size=4, stride=1, padding=0, bias=False),  # 4x4 -> 1x1
            ]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def set_requires_grad(self, grad):
        for param in self.parameters():
            param.requires_grad = grad

    def forward(self, input):
        return self.network(input).reshape(input.size(0), -1)


if __name__ == '__main__':
    netD = EDDeconv(cin=3, cout=1, nf=64, zdim=256, activation=None)
    netA = EDDeconv(cin=3, cout=3, nf=64, zdim=256)
    netL = Encoder(cin=3, cout=4, nf=32)
    netV = Encoder(cin=3, cout=6, nf=32)
    netC = ConfNet(cin=3, cout=2, nf=64, zdim=128)

    inputs = torch.randn((2, 3, 64, 64))

    print(netD(inputs).shape)
    print(netA(inputs).shape)
    print(netL(inputs).shape)
    print(netV(inputs).shape)
    a,b= netC(inputs)
    print(a.shape,b.shape)