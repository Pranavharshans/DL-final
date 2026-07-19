"""20 model architectures — each <= 5M parameters, 18-class country classifier."""
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 18

MODEL_REGISTRY = {}

def register(name):
    def decorator(fn):
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============================================================
# Shared building blocks
# ============================================================

def conv3x3(in_c, out_c, stride=1, padding=1):
    return nn.Conv2d(in_c, out_c, 3, stride, padding, bias=False)

def conv1x1(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 1, stride, bias=False)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = conv3x3(in_c, out_c, stride)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DepthwiseSepConv(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False)
        self.pw = conv1x1(in_c, out_c)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn2(self.pw(self.act(self.bn1(self.dw(x))))))

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_c, out_c, stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = conv3x3(out_c, out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(conv1x1(in_c, out_c, stride), nn.BatchNorm2d(out_c))
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)

class SEResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, reduction=16):
        super().__init__()
        self.conv1 = conv3x3(in_c, out_c, stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = conv3x3(out_c, out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        se_mid = max(1, out_c // reduction)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), conv1x1(out_c, se_mid),
                                nn.ReLU(inplace=True), conv1x1(se_mid, out_c), nn.Sigmoid())
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(conv1x1(in_c, out_c, stride), nn.BatchNorm2d(out_c))
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out * self.se(out) + identity)

class CBAMBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, reduction=16):
        super().__init__()
        self.conv1 = conv3x3(in_c, out_c, stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = conv3x3(out_c, out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        se_mid = max(1, out_c // reduction)
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), conv1x1(out_c, se_mid),
                                nn.ReLU(inplace=True), conv1x1(se_mid, out_c), nn.Sigmoid())
        self.sa = nn.Sequential(nn.Conv2d(2, 1, 7, padding=3, bias=False), nn.Sigmoid())
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(conv1x1(in_c, out_c, stride), nn.BatchNorm2d(out_c))
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out * self.ca(out)
        sp = torch.cat([out.mean(dim=1, keepdim=True), out.max(dim=1, keepdim=True)[0]], dim=1)
        out = out * self.sa(sp)
        return self.act(out + identity)

class DenseLayer(nn.Module):
    def __init__(self, in_c, growth_rate):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_c)
        self.conv = conv3x3(in_c, growth_rate, padding=1)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return torch.cat([x, self.act(self.conv(self.bn(x)))], dim=1)

# ============================================================
# MODEL 1: CustomCNN-Small (~1.6M)
# ============================================================
@register("01_CustomCNN-Small")
def model_01():
    return nn.Sequential(
        ConvBlock(3, 32), nn.MaxPool2d(2),
        ConvBlock(32, 64), nn.MaxPool2d(2),
        ConvBlock(64, 128), nn.MaxPool2d(2),
        ConvBlock(128, 256), nn.MaxPool2d(2),
        ConvBlock(256, 512), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Linear(512, NUM_CLASSES),
    )

# ============================================================
# MODEL 2: CustomCNN-Medium (~3M)
# ============================================================
@register("02_CustomCNN-Medium")
def model_02():
    return nn.Sequential(
        ConvBlock(3, 48), ConvBlock(48, 48), nn.MaxPool2d(2),
        ConvBlock(48, 96), ConvBlock(96, 96), nn.MaxPool2d(2),
        ConvBlock(96, 192), ConvBlock(192, 192), nn.MaxPool2d(2),
        ConvBlock(192, 384), ConvBlock(384, 384), nn.MaxPool2d(2),
        ConvBlock(384, 512), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(512, NUM_CLASSES),
    )

# ============================================================
# MODEL 3: CustomCNN-Large (~4.5M)
# ============================================================
@register("03_CustomCNN-Large")
def model_03():
    return nn.Sequential(
        ConvBlock(3, 64), ConvBlock(64, 64), nn.MaxPool2d(2),
        ConvBlock(64, 128), ConvBlock(128, 128), nn.MaxPool2d(2),
        ConvBlock(128, 192), ConvBlock(192, 192), nn.MaxPool2d(2),
        ConvBlock(192, 256), ConvBlock(256, 256), nn.MaxPool2d(2),
        ConvBlock(256, 384), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(384, NUM_CLASSES),
    )

# ============================================================
# MODEL 4: MiniResNet (~4M) — 3 stages, fewer channels
# ============================================================
class MiniResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(conv3x3(3, 32), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(ResBlock(32, 64), ResBlock(64, 64))
        self.layer2 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128))
        self.layer3 = nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, NUM_CLASSES)
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return self.fc(torch.flatten(self.pool(x), 1))

@register("04_MiniResNet")
def model_04():
    return MiniResNet()

# ============================================================
# MODEL 5: MobileNet-Style (~1.4M)
# ============================================================
class MobileStyleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, 2),
            DepthwiseSepConv(32, 64),
            DepthwiseSepConv(64, 128, 2),
            DepthwiseSepConv(128, 128),
            DepthwiseSepConv(128, 256, 2),
            DepthwiseSepConv(256, 256),
            DepthwiseSepConv(256, 512, 2),
            DepthwiseSepConv(512, 512),
            DepthwiseSepConv(512, 512),
            DepthwiseSepConv(512, 1024, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(1024, NUM_CLASSES))
    def forward(self, x):
        return self.fc(torch.flatten(self.features(x), 1))

@register("05_MobileNet-Style")
def model_05():
    return MobileStyleNet()

# ============================================================
# MODEL 6: ShuffleNet-Style (~3.5M)
# ============================================================
class ShuffleBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        mid = out_c // 2
        self.b1 = nn.Sequential(conv1x1(in_c, mid), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
                                nn.Conv2d(mid, mid, 3, 2, 1, groups=mid, bias=False),
                                nn.BatchNorm2d(mid), conv1x1(mid, out_c - in_c), nn.BatchNorm2d(out_c - in_c),
                                nn.ReLU(inplace=True))
        self.shortcut = nn.Sequential(nn.Conv2d(in_c, in_c, 3, 2, 1, groups=in_c, bias=False),
                                      nn.BatchNorm2d(in_c), conv1x1(in_c, out_c - in_c),
                                      nn.BatchNorm2d(out_c - in_c))
    def forward(self, x):
        s = self.shortcut(x)
        r = self.b1(x)
        out = torch.cat([s, r], dim=1)
        b, c, h, w = out.shape
        return out.view(b, 2, c//2, h, w).transpose(1,2).contiguous().view(b,c,h,w)

class ShuffleStyleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(3, 64, 2), nn.MaxPool2d(3,2,1))
        self.stg2 = nn.Sequential(ShuffleBlock(64, 128), ShuffleBlock(128, 256))
        self.stg3 = nn.Sequential(ShuffleBlock(256, 512), ShuffleBlock(512, 1024))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, NUM_CLASSES)
    def forward(self, x):
        x = self.stem(x); x = self.stg2(x); x = self.stg3(x)
        return self.fc(torch.flatten(self.pool(x), 1))

@register("06_ShuffleNet-Style")
def model_06():
    return ShuffleStyleNet()

# ============================================================
# MODEL 7: SE-ResNet (~3.5M) — fewer channels
# ============================================================
class SEResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(conv3x3(3, 32), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(SEResBlock(32, 64), SEResBlock(64, 64))
        self.layer2 = nn.Sequential(SEResBlock(64, 128, 2), SEResBlock(128, 128))
        self.layer3 = nn.Sequential(SEResBlock(128, 256, 2), SEResBlock(256, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, NUM_CLASSES)
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return self.fc(torch.flatten(self.pool(x), 1))

@register("07_SE-ResNet")
def model_07():
    return SEResNet()

# ============================================================
# MODEL 8: CBAM-Net (~3.5M)
# ============================================================
class CBAMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(conv3x3(3, 32), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.l1 = nn.Sequential(CBAMBlock(32, 64), CBAMBlock(64, 64))
        self.l2 = nn.Sequential(CBAMBlock(64, 128, 2), CBAMBlock(128, 128))
        self.l3 = nn.Sequential(CBAMBlock(128, 256, 2), CBAMBlock(256, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, NUM_CLASSES)
    def forward(self, x):
        x = self.stem(x); x = self.l1(x); x = self.l2(x); x = self.l3(x)
        return self.fc(torch.flatten(self.pool(x), 1))

@register("08_CBAM-Net")
def model_08():
    return CBAMNet()

# ============================================================
# MODEL 9: ConvNeXt-Micro (~4.5M)
# ============================================================
class CNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)
    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pw2(self.act(self.pw1(self.norm(x))))
        x = x.permute(0, 3, 1, 2)
        return identity + self.gamma.view(1, -1, 1, 1) * x

class ConvNeXtMicro(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [48, 96, 192, 320]
        self.stem = nn.Sequential(nn.Conv2d(3, dims[0], 4, 4), nn.BatchNorm2d(dims[0]))
        self.s1 = nn.Sequential(*[CNeXtBlock(dims[0]) for _ in range(2)])
        self.d1 = nn.Sequential(nn.LayerNorm(dims[0]), nn.Linear(dims[0], dims[1]))
        self.s2 = nn.Sequential(*[CNeXtBlock(dims[1]) for _ in range(2)])
        self.d2 = nn.Sequential(nn.LayerNorm(dims[1]), nn.Linear(dims[1], dims[2]))
        self.s3 = nn.Sequential(*[CNeXtBlock(dims[2]) for _ in range(3)])
        self.d3 = nn.Sequential(nn.LayerNorm(dims[2]), nn.Linear(dims[2], dims[3]))
        self.s4 = nn.Sequential(*[CNeXtBlock(dims[3]) for _ in range(2)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dims[3], NUM_CLASSES)
    def forward(self, x):
        x = self.stem(x)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1); x = self.d1(self.s1(x)); x = x.permute(0, 3, 1, 2)
        x = x.permute(0, 2, 3, 1); x = self.d2(self.s2(x)); x = x.permute(0, 3, 1, 2)
        x = x.permute(0, 2, 3, 1); x = self.d3(self.s3(x)); x = x.permute(0, 3, 1, 2)
        x = self.s4(x)
        return self.fc(torch.flatten(self.pool(x), 1))

@register("09_ConvNeXt-Micro")
def model_09():
    return ConvNeXtMicro()

# ============================================================
# MODEL 10: MultiScale-CNN (~3M)
# ============================================================
class MSBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        mid = out_c // 4
        self.b1 = nn.Sequential(conv1x1(in_c, mid), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(conv1x1(in_c, mid), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
                                conv3x3(mid, mid), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.b5 = nn.Sequential(conv1x1(in_c, mid), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
                                conv3x3(mid, mid), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
                                conv3x3(mid, mid), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.mp = nn.Sequential(nn.MaxPool2d(3,1,1), conv1x1(in_c, mid), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.mp(x)], dim=1)

class MultiScaleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(3, 64, 2), nn.MaxPool2d(3,2,1))
        self.ms1 = nn.Sequential(MSBlock(64, 128), conv1x1(128, 128))
        self.ms2 = nn.Sequential(MSBlock(128, 256), conv1x1(256, 256))
        self.ms3 = nn.Sequential(MSBlock(256, 512), conv1x1(512, 512))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, NUM_CLASSES))
    def forward(self, x):
        x = self.stem(x); x = self.ms1(x); x = self.ms2(x); x = self.ms3(x)
        return self.fc(torch.flatten(self.pool(x), 1))

@register("10_MultiScale-CNN")
def model_10():
    return MultiScaleCNN()

# ============================================================
# MODEL 11: DenseNet-Style (~2.5M)
# ============================================================
class DenseStyleNet(nn.Module):
    def __init__(self, gr=32):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(3, 64, 2), nn.MaxPool2d(3,2,1))
        self.d1, c1 = self._make_dense(64, gr, 4)
        self.t1 = self._trans(c1, 128)
        self.d2, c2 = self._make_dense(128, gr, 4)
        self.t2 = self._trans(c2, 256)
        self.d3, c3 = self._make_dense(256, gr, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c3, NUM_CLASSES)
    def _make_dense(self, in_c, gr, n):
        layers, c = [], in_c
        for _ in range(n):
            layers.append(DenseLayer(c, gr)); c += gr
        return nn.Sequential(*layers), c
    def _trans(self, in_c, out_c):
        return nn.Sequential(nn.BatchNorm2d(in_c), conv1x1(in_c, out_c), nn.AvgPool2d(2))
    def forward(self, x):
        x = self.stem(x); x = self.d1(x); x = self.t1(x)
        x = self.d2(x); x = self.t2(x); x = self.d3(x)
        return self.fc(torch.flatten(self.pool(x), 1))

@register("11_DenseNet-Style")
def model_11():
    return DenseStyleNet()

# ============================================================
# MODEL 12: Wide-ResNet (~4.5M)
# ============================================================
class WideResNet(nn.Module):
    def __init__(self, w=3):
        super().__init__()
        c = lambda x: x * w
        self.stem = nn.Sequential(conv3x3(3, c(16)), nn.BatchNorm2d(c(16)), nn.ReLU(inplace=True))
        self.l1 = nn.Sequential(ResBlock(c(16), c(32)), ResBlock(c(32), c(32)))
        self.l2 = nn.Sequential(ResBlock(c(32), c(64), 2), ResBlock(c(64), c(64)))
        self.l3 = nn.Sequential(ResBlock(c(64), c(128), 2), ResBlock(c(128), c(128)))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c(128), NUM_CLASSES)
    def forward(self, x):
        x = self.stem(x); x = self.l1(x); x = self.l2(x); x = self.l3(x)
        return self.fc(torch.flatten(self.pool(x), 1))

@register("12_Wide-ResNet")
def model_12():
    return WideResNet(w=2)

# ============================================================
# MODEL 13: CNN+SelfAttention (~3.5M)
# ============================================================
class SA2D(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1).transpose(1, 2)
        qkv = self.qkv(x).reshape(b, h*w, 3, self.heads, c//self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = (attn.softmax(dim=-1) @ v).transpose(1,2).reshape(b, h*w, c)
        return self.proj(x).transpose(1,2).view(b, c, h, w)

class CNNSA(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(3, 32, 2), nn.MaxPool2d(3,2,1))
        self.l1 = nn.Sequential(ResBlock(32, 64, 2), ResBlock(64, 64))
        self.l2 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128))
        self.attn = SA2D(128, heads=4)
        self.l3 = nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, NUM_CLASSES))
    def forward(self, x):
        x = self.stem(x); x = self.l1(x); x = self.l2(x)
        x = self.attn(x); x = self.l3(x)
        return self.fc(torch.flatten(self.pool(x), 1))

@register("13_CNN+SelfAttention")
def model_13():
    return CNNSA()

# ============================================================
# MODEL 14: CNN+Transformer (~4.5M)
# ============================================================
class TfBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(), nn.Linear(dim*2, dim))
    def forward(self, x):
        x = x + self.attn(self.n1(x), self.n1(x), self.n1(x))[0]
        return x + self.mlp(self.n2(x))

class CNNTf(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(3, 32, 2), nn.MaxPool2d(3,2,1))
        self.l1 = nn.Sequential(ResBlock(32, 64, 2), ResBlock(64, 64))
        self.l2 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128))
        self.l3 = nn.Sequential(ResBlock(128, 192, 2), ResBlock(192, 192))
        self.tf = nn.Sequential(*[TfBlock(192, heads=4) for _ in range(3)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(192, NUM_CLASSES))
    def forward(self, x):
        x = self.stem(x); x = self.l1(x); x = self.l2(x); x = self.l3(x)
        b, c, h, w = x.shape
        x = x.view(b, c, -1).transpose(1, 2)
        x = self.tf(x)
        x = x.transpose(1, 2).view(b, c, h, w)
        return self.fc(torch.flatten(self.pool(x), 1))

@register("14_CNN+Transformer")
def model_14():
    return CNNTf()

# ============================================================
# MODEL 15: GeoDualBranch (~3.5M)
# ============================================================
class GeoDualBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(3, 32, 2), nn.MaxPool2d(3,2,1))
        self.l1 = nn.Sequential(ResBlock(32, 64, 2), ResBlock(64, 64))
        self.l2 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128))
        self.l3 = nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.img_fc = nn.Linear(256, 192)
        self.geo_net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        self.fusion = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, NUM_CLASSES))
    def forward(self, x, coords):
        f = torch.flatten(self.pool(self.l3(self.l2(self.l1(self.stem(x))))), 1)
        img = F.relu(self.img_fc(f))
        geo = F.relu(self.geo_net(coords))
        return self.fusion(torch.cat([img, geo], dim=1))

@register("15_GeoDualBranch")
def model_15():
    return GeoDualBranch()

# ============================================================
# MODEL 16: GeoFiLM (~3M)
# ============================================================
class FiLMBlock(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.conv1 = conv3x3(in_c, in_c)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv2 = conv3x3(in_c, in_c)
        self.bn2 = nn.BatchNorm2d(in_c)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x, gamma, beta):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out * gamma.view(-1, out.size(1), 1, 1) + beta.view(-1, out.size(1), 1, 1) + x)

class GeoFiLMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.geo_enc = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 256))
        self.stem = nn.Sequential(ConvBlock(3, 32, 2), nn.MaxPool2d(3,2,1))
        self.f1 = FiLMBlock(32)
        self.d1 = ConvBlock(32, 64, 2)
        self.f2 = FiLMBlock(64)
        self.d2 = ConvBlock(64, 128, 2)
        self.f3 = FiLMBlock(128)
        self.d3 = ConvBlock(128, 256, 2)
        self.f4 = FiLMBlock(256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, NUM_CLASSES))
        self.g_nets = nn.ModuleList([nn.Linear(256, 32), nn.Linear(256, 64), nn.Linear(256, 128), nn.Linear(256, 256)])
        self.b_nets = nn.ModuleList([nn.Linear(256, 32), nn.Linear(256, 64), nn.Linear(256, 128), nn.Linear(256, 256)])
    def forward(self, x, coords):
        geo = self.geo_enc(coords)
        g = [gn(geo) for gn in self.g_nets]
        b = [bn(geo) for bn in self.b_nets]
        x = self.stem(x); x = self.f1(x, g[0], b[0]); x = self.d1(x)
        x = self.f2(x, g[1], b[1]); x = self.d2(x)
        x = self.f3(x, g[2], b[2]); x = self.d3(x)
        x = self.f4(x, g[3], b[3])
        return self.fc(torch.flatten(self.pool(x), 1))

@register("16_GeoFiLM")
def model_16():
    return GeoFiLMNet()

# ============================================================
# MODEL 17: MultiTask-Geo (~3M)
# ============================================================
class MultiTaskGeo(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(3, 32, 2), nn.MaxPool2d(3,2,1))
        self.l1 = nn.Sequential(ResBlock(32, 64, 2), ResBlock(64, 64))
        self.l2 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128))
        self.l3 = nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.country_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, NUM_CLASSES))
        self.coord_head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2))
    def forward(self, x, coords=None):
        f = torch.flatten(self.pool(self.l3(self.l2(self.l1(self.stem(x))))), 1)
        return self.country_head(f), self.coord_head(f)

@register("17_MultiTask-Geo")
def model_17():
    return MultiTaskGeo()

# ============================================================
# MODEL 18: EfficientNet-Style (~4M)
# ============================================================
class MBConv(nn.Module):
    def __init__(self, in_c, out_c, expand=4, stride=1, se_ratio=4):
        super().__init__()
        mid = in_c * expand
        self.use_res = stride == 1 and in_c == out_c
        self.exp = nn.Sequential(conv1x1(in_c, mid), nn.BatchNorm2d(mid), nn.SiLU()) if expand > 1 else nn.Identity()
        self.dw = nn.Sequential(nn.Conv2d(mid if expand>1 else in_c, mid if expand>1 else in_c, 3, stride, 1, groups=mid if expand>1 else in_c, bias=False),
                                nn.BatchNorm2d(mid if expand>1 else in_c), nn.SiLU())
        se_mid = max(1, (mid if expand>1 else in_c) // se_ratio)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), conv1x1(mid if expand>1 else in_c, se_mid),
                                nn.SiLU(), conv1x1(se_mid, mid if expand>1 else in_c), nn.Sigmoid())
        self.proj = nn.Sequential(conv1x1(mid if expand>1 else in_c, out_c), nn.BatchNorm2d(out_c))
    def forward(self, x):
        identity = x
        x = self.exp(x); x = self.dw(x); x = x * self.se(x); x = self.proj(x)
        return x + identity if self.use_res else x

class EfficientStyleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(3, 32, 2))
        self.blocks = nn.Sequential(
            MBConv(32, 16, expand=1),
            MBConv(16, 24, expand=4, stride=2), MBConv(24, 24, expand=4),
            MBConv(24, 40, expand=4, stride=2), MBConv(40, 40, expand=4),
            MBConv(40, 80, expand=6, stride=2), MBConv(80, 80, expand=6),
            MBConv(80, 112, expand=6), MBConv(112, 112, expand=6),
            MBConv(112, 192, expand=6, stride=2), MBConv(192, 192, expand=6),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(192, NUM_CLASSES))
    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.blocks(self.stem(x))), 1))

@register("18_EfficientNet-Style")
def model_18():
    return EfficientStyleNet()

# ============================================================
# MODEL 19: RepVGG-Style (~4M)
# ============================================================
class RepVGGBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.b3 = nn.Sequential(conv3x3(in_c, out_c, stride), nn.BatchNorm2d(out_c))
        self.b1 = nn.Sequential(conv1x1(in_c, out_c, stride), nn.BatchNorm2d(out_c))
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.b3(x) + self.b1(x))

class RepVGGStyleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.Sequential(
            RepVGGBlock(3, 64, 2), RepVGGBlock(64, 64),
            RepVGGBlock(64, 96, 2), RepVGGBlock(96, 96),
            RepVGGBlock(96, 160, 2), RepVGGBlock(160, 160), RepVGGBlock(160, 160),
            RepVGGBlock(160, 320, 2), RepVGGBlock(320, 320),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(320, NUM_CLASSES)
    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.stages(x)), 1))

@register("19_RepVGG-Style")
def model_19():
    return RepVGGStyleNet()

# ============================================================
# MODEL 20: CompactNet (~800K)
# ============================================================
class CompactNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            DepthwiseSepConv(3, 32, 2),
            DepthwiseSepConv(32, 64),
            DepthwiseSepConv(64, 128, 2),
            DepthwiseSepConv(128, 128),
            DepthwiseSepConv(128, 256, 2),
            DepthwiseSepConv(256, 256),
            DepthwiseSepConv(256, 512, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, NUM_CLASSES))
    def forward(self, x):
        return self.fc(torch.flatten(self.features(x), 1))

@register("20_CompactNet")
def model_20():
    return CompactNet()
