"""20 model architectures — each ≤ 5M parameters, 18-class country classifier."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 18

# ============================================================
# Model registry
# ============================================================

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

def conv3x3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)

def conv1x1(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 1, stride, bias=False)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, use_bn=True):
        super().__init__()
        self.conv = conv3x3(in_c, out_c, stride)
        self.bn = nn.BatchNorm2d(out_c) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DepthwiseSepConv(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, 3, stride, 1, groups=in_c, bias=False)
        self.pointwise = conv1x1(in_c, out_c)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x

class SEResBlock(nn.Module):
    """Residual block with Squeeze-and-Excitation."""
    def __init__(self, in_c, out_c, stride=1, reduction=16):
        super().__init__()
        self.conv1 = conv3x3(in_c, out_c, stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = conv3x3(out_c, out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(out_c, out_c // reduction),
            nn.ReLU(inplace=True),
            conv1x1(out_c // reduction, out_c),
            nn.Sigmoid(),
        )
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(conv1x1(in_c, out_c, stride), nn.BatchNorm2d(out_c))
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out * self.se(out)
        out += identity
        return self.act(out)

class CBAMBlock(nn.Module):
    """Convolutional Block Attention Module."""
    def __init__(self, in_c, out_c, stride=1, reduction=16):
        super().__init__()
        self.conv1 = conv3x3(in_c, out_c, stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = conv3x3(out_c, out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        # Channel attention
        self.ch_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(out_c, out_c // reduction),
            nn.ReLU(inplace=True),
            conv1x1(out_c // reduction, out_c),
        )
        # Spatial attention
        self.sp_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(conv1x1(in_c, out_c, stride), nn.BatchNorm2d(out_c))
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Channel attention
        ch = self.ch_att(out)
        ch = torch.sigmoid(ch + self.ch_att(torch.max(out, dim=1, keepdim=True)[0].mean(dim=(2,3), keepdim=True).expand(-1, out.size(1), -1, -1)))
        out = out * ch
        # Spatial attention
        sp = torch.cat([out.mean(dim=1, keepdim=True), out.max(dim=1, keepdim=True)[0]], dim=1)
        out = out * self.sp_att(sp)
        out += identity
        return self.act(out)

class ResBlock(nn.Module):
    """Standard residual block."""
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

class DenseLayer(nn.Module):
    def __init__(self, in_c, growth_rate):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_c)
        self.conv = conv3x3(in_c, growth_rate)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return torch.cat([x, self.act(self.conv(self.bn(x)))], dim=1)


# ============================================================
# MODEL 1: CustomCNN-Small (~500K)
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
# MODEL 2: CustomCNN-Medium (~1.5M)
# ============================================================
@register("02_CustomCNN-Medium")
def model_02():
    return nn.Sequential(
        ConvBlock(3, 64), ConvBlock(64, 64), nn.MaxPool2d(2),
        ConvBlock(64, 128), ConvBlock(128, 128), nn.MaxPool2d(2),
        ConvBlock(128, 256), ConvBlock(256, 256), nn.MaxPool2d(2),
        ConvBlock(256, 512), ConvBlock(512, 512), nn.MaxPool2d(2),
        ConvBlock(512, 1024), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(1024, NUM_CLASSES),
    )

# ============================================================
# MODEL 3: CustomCNN-Large (~3.5M)
# ============================================================
@register("03_CustomCNN-Large")
def model_03():
    return nn.Sequential(
        ConvBlock(3, 64), ConvBlock(64, 64), nn.MaxPool2d(2),
        ConvBlock(64, 128), ConvBlock(128, 128), ConvBlock(128, 128), nn.MaxPool2d(2),
        ConvBlock(128, 256), ConvBlock(256, 256), ConvBlock(256, 256), nn.MaxPool2d(2),
        ConvBlock(256, 512), ConvBlock(512, 512), ConvBlock(512, 512), nn.MaxPool2d(2),
        ConvBlock(512, 1024), ConvBlock(1024, 1024), nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(1024, NUM_CLASSES),
    )

# ============================================================
# MODEL 4: MiniResNet (~2.5M)
# ============================================================
class MiniResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(conv3x3(3, 64), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(ResBlock(64, 64), ResBlock(64, 64))
        self.layer2 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128))
        self.layer3 = nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256))
        self.layer4 = nn.Sequential(ResBlock(256, 512, 2), ResBlock(512, 512))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, NUM_CLASSES)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.fc(torch.flatten(self.pool(x), 1))

@register("04_MiniResNet")
def model_04():
    return MiniResNet()

# ============================================================
# MODEL 5: MobileNet-Style (~1.8M)
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
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(1024, NUM_CLASSES))
    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.features(x)), 1))

@register("05_MobileNet-Style")
def model_05():
    return MobileStyleNet()

# ============================================================
# MODEL 6: ShuffleNet-Style (~2M)
# ============================================================
class ShuffleBlock(nn.Module):
    def __init__(self, in_c, out_c, groups=4):
        super().__init__()
        mid = out_c // 2
        self.conv1 = nn.Conv2d(in_c, mid, 1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.dw = nn.Conv2d(mid, mid, 3, 1, 1, groups=mid, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, out_c - in_c, 1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c - in_c)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.dw(out))
        # Channel shuffle
        b, c, h, w = out.shape
        g = 4
        out = out.view(b, g, c//g, h, w).transpose(1,2).contiguous().view(b,c,h,w)
        out = self.bn3(self.conv2(out))
        return torch.cat([x, self.act(out)], dim=1)

class ShuffleStyleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(3, 64, 2), nn.MaxPool2d(3,2,1))
        self.stage2 = nn.Sequential(ShuffleBlock(64, 128), ShuffleBlock(128, 128))
        self.stage3 = nn.Sequential(ShuffleBlock(128, 256), ShuffleBlock(256, 256))
        self.stage4 = nn.Sequential(ShuffleBlock(256, 512), ShuffleBlock(512, 512))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, NUM_CLASSES)
    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x); x = self.stage3(x); x = self.stage4(x)
        return self.fc(torch.flatten(self.pool(x), 1))

@register("06_ShuffleNet-Style")
def model_06():
    return ShuffleStyleNet()

# ============================================================
# MODEL 7: SE-ResNet (~3.5M)
# ============================================================
class SEResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(conv3x3(3, 64), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(SEResBlock(64, 64), SEResBlock(64, 64))
        self.layer2 = nn.Sequential(SEResBlock(64, 128, 2), SEResBlock(128, 128))
        self.layer3 = nn.Sequential(SEResBlock(128, 256, 2), SEResBlock(256, 256))
        self.layer4 = nn.Sequential(SEResBlock(256, 512, 2), SEResBlock(512, 512))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, NUM_CLASSES)
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
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
        self.stem = nn.Sequential(conv3x3(3, 64), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(CBAMBlock(64, 64), CBAMBlock(64, 64))
        self.layer2 = nn.Sequential(CBAMBlock(64, 128, 2), CBAMBlock(128, 128))
        self.layer3 = nn.Sequential(CBAMBlock(128, 256, 2), CBAMBlock(256, 256))
        self.layer4 = nn.Sequential(CBAMBlock(256, 512, 2), CBAMBlock(512, 512))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, NUM_CLASSES)
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.fc(torch.flatten(self.pool(x), 1))

@register("08_CBAM-Net")
def model_08():
    return CBAMNet()

# ============================================================
# MODEL 9: ConvNeXt-Tiny (~4.5M)
# ============================================================
class ConvNeXtBlock(nn.Module):
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
        x = self.norm(x)
        x = self.pw2(self.act(self.pw1(x)))
        x = x.permute(0, 3, 1, 2)
        return identity + self.gamma.view(1, -1, 1, 1) * x

class ConvNeXtTiny(nn.Module):
    def __init__(self):
        super().__init__()
        dims = [96, 192, 384, 768]
        self.stem = nn.Sequential(nn.Conv2d(3, dims[0], 4, 4), nn.LayerNorm([dims[0], 128, 128]))
        self.stages = nn.ModuleList()
        for i in range(4):
            blocks = []
            for _ in range([3,3,9,3][i]):
                blocks.append(ConvNeXtBlock(dims[i]))
            self.stages.append(nn.Sequential(*blocks))
            if i < 3:
                self.stages.append(nn.Sequential(nn.LayerNorm([dims[i], 128>>i, 128>>i]),
                                                  nn.Linear(dims[i], dims[i+1]),
                                                  nn.GELU()))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dims[-1], NUM_CLASSES)
    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return self.fc(torch.flatten(self.pool(x), 1))

@register("09_ConvNeXt-Tiny")
def model_09():
    return ConvNeXtTiny()

# ============================================================
# MODEL 10: MultiScale-CNN (~2.5M)
# ============================================================
class MultiScaleBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        mid = out_c // 4
        self.b1 = nn.Sequential(conv1x1(in_c, mid), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(conv1x1(in_c, mid), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
                                 conv3x3(mid, mid, padding=1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.b5 = nn.Sequential(conv1x1(in_c, mid), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
                                 conv3x3(mid, mid, padding=1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
                                 conv3x3(mid, mid, padding=1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.pool = nn.Sequential(nn.MaxPool2d(3,1,1), conv1x1(in_c, mid), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.pool(x)], dim=1)

class MultiScaleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(3, 64, 2), nn.MaxPool2d(3,2,1))
        self.ms1 = MultiScaleBlock(64, 128)
        self.trans1 = conv1x1(128, 128)
        self.ms2 = MultiScaleBlock(128, 256)
        self.trans2 = conv1x1(256, 256)
        self.ms3 = MultiScaleBlock(256, 512)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, NUM_CLASSES))
    def forward(self, x):
        x = self.stem(x)
        x = self.trans1(self.ms1(x))
        x = self.trans2(self.ms2(x))
        x = self.ms3(x)
        return self.fc(torch.flatten(self.pool(x), 1))

@register("10_MultiScale-CNN")
def model_10():
    return MultiScaleCNN()

# ============================================================
# MODEL 11: DenseNet-Style (~2.5M)
# ============================================================
class DenseStyleNet(nn.Module):
    def __init__(self, growth_rate=32):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(3, 64, 2), nn.MaxPool2d(3,2,1))
        # Dense blocks
        self.dense1 = self._make_dense(64, growth_rate, 4)
        self.trans1 = self._transition(64 + 4*growth_rate, 128)
        self.dense2 = self._make_dense(128, growth_rate, 4)
        self.trans2 = self._transition(128 + 4*growth_rate, 256)
        self.dense3 = self._make_dense(256, growth_rate, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256 + 4*growth_rate, NUM_CLASSES)
    def _make_dense(self, in_c, gr, n):
        layers = []
        for _ in range(n):
            layers.append(DenseLayer(in_c, gr))
            in_c += gr
        return nn.Sequential(*layers)
    def _transition(self, in_c, out_c):
        return nn.Sequential(nn.BatchNorm2d(in_c), conv1x1(in_c, out_c), nn.AvgPool2d(2))
    def forward(self, x):
        x = self.stem(x); x = self.dense1(x); x = self.trans1(x)
        x = self.dense2(x); x = self.trans2(x); x = self.dense3(x)
        return self.fc(torch.flatten(self.pool(x), 1))

@register("11_DenseNet-Style")
def model_11():
    return DenseStyleNet()

# ============================================================
# MODEL 12: Wide-ResNet (~4.5M)
# ============================================================
class WideResNet(nn.Module):
    def __init__(self, widen=4):
        super().__init__()
        w = lambda c: c * widen
        self.stem = nn.Sequential(conv3x3(3, w(16)), nn.BatchNorm2d(w(16)), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(ResBlock(w(16), w(32)), ResBlock(w(32), w(32)))
        self.layer2 = nn.Sequential(ResBlock(w(32), w(64), 2), ResBlock(w(64), w(64)))
        self.layer3 = nn.Sequential(ResBlock(w(64), w(128), 2), ResBlock(w(128), w(128)))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w(128), NUM_CLASSES)
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        return self.fc(torch.flatten(self.pool(x), 1))

@register("12_Wide-ResNet")
def model_12():
    return WideResNet(widen=4)

# ============================================================
# MODEL 13: CNN+SelfAttention (~3.5M)
# ============================================================
class SelfAttention2D(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1).transpose(1, 2)  # B, N, C
        qkv = self.qkv(x).reshape(b, h*w, 3, self.heads, c//self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1,2).reshape(b, h*w, c)
        x = self.proj(x).transpose(1,2).view(b, c, h, w)
        return x

class CNNSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(3, 64, 2), nn.MaxPool2d(3,2,1))
        self.layer1 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128))
        self.layer2 = nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256))
        self.attn = SelfAttention2D(256, heads=4)
        self.layer3 = nn.Sequential(ResBlock(256, 512, 2), ResBlock(512, 512))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, NUM_CLASSES))
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x)
        x = self.attn(x)
        x = self.layer3(x)
        return self.fc(torch.flatten(self.pool(x), 1))

@register("13_CNN+SelfAttention")
def model_13():
    return CNNSelfAttention()

# ============================================================
# MODEL 14: CNN+Transformer (~4.5M)
# ============================================================
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*mlp_ratio), nn.GELU(), nn.Linear(dim*mlp_ratio, dim))
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class CNNTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(3, 64, 2), nn.MaxPool2d(3,2,1))
        self.layer1 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128))
        self.layer2 = nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256))
        self.layer3 = nn.Sequential(ResBlock(256, 384, 2), ResBlock(384, 384))
        self.tf_blocks = nn.Sequential(*[TransformerBlock(384, heads=4) for _ in range(2)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(384, NUM_CLASSES))
    def forward(self, x):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        b, c, h, w = x.shape
        x = x.view(b, c, -1).transpose(1, 2)
        x = self.tf_blocks(x)
        x = x.transpose(1, 2).view(b, c, h, w)
        return self.fc(torch.flatten(self.pool(x), 1))

@register("14_CNN+Transformer")
def model_14():
    return CNNTransformer()

# ============================================================
# MODEL 15: GeoDualBranch (~3.5M)
# ============================================================
class GeoDualBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # Image branch
        self.stem = nn.Sequential(ConvBlock(3, 64, 2), nn.MaxPool2d(3,2,1))
        self.layer1 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128))
        self.layer2 = nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256))
        self.layer3 = nn.Sequential(ResBlock(256, 512, 2), ResBlock(512, 512))
        self.img_pool = nn.AdaptiveAvgPool2d(1)
        self.img_fc = nn.Linear(512, 256)
        # Geo branch
        self.geo_net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 128))
        # Fusion
        self.fusion = nn.Sequential(nn.Linear(256+128, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, NUM_CLASSES))
    def forward(self, x, coords):
        img = torch.flatten(self.img_pool(self.layer3(self.layer2(self.layer1(self.stem(x))))), 1)
        img = F.relu(self.img_fc(img))
        geo = F.relu(self.geo_net(coords))
        return self.fusion(torch.cat([img, geo], dim=1))

@register("15_GeoDualBranch")
def model_15():
    return GeoDualBranch()

# ============================================================
# MODEL 16: GeoFiLM (~3M)
# ============================================================
class FiLMBlock(nn.Module):
    def __init__(self, in_c, geo_dim=128):
        super().__init__()
        self.conv1 = conv3x3(in_c, in_c)
        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv2 = conv3x3(in_c, in_c)
        self.bn2 = nn.BatchNorm2d(in_c)
        self.gamma_net = nn.Linear(geo_dim, in_c)
        self.beta_net = nn.Linear(geo_dim, in_c)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x, gamma, beta):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out * gamma.view(-1, out.size(1), 1, 1) + beta.view(-1, out.size(1), 1, 1)
        return self.act(out + x)

class GeoFiLMNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.geo_enc = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 128))
        self.stem = nn.Sequential(ConvBlock(3, 64, 2), nn.MaxPool2d(3,2,1))
        self.film1 = FiLMBlock(64, 128)
        self.down1 = nn.Sequential(ConvBlock(64, 128, 2))
        self.film2 = FiLMBlock(128, 128)
        self.down2 = nn.Sequential(ConvBlock(128, 256, 2))
        self.film3 = FiLMBlock(256, 128)
        self.down3 = nn.Sequential(ConvBlock(256, 512, 2))
        self.film4 = FiLMBlock(512, 128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, NUM_CLASSES))
    def forward(self, x, coords):
        geo = self.geo_enc(coords)
        gamma = geo[:, :64]
        beta = geo[:, 64:]
        # Pad FiLM params to match channel counts
        g1 = F.pad(gamma, (0, 64-64)) if 64 < 64 else gamma[:, :64]
        b1 = F.pad(beta, (0, 64-64)) if 64 < 64 else beta[:, :64]
        g2, b2 = gamma[:, :128], beta[:, :128]
        g3, b3 = gamma[:128], beta[:128]
        # Actually use consistent geo dims
        x = self.stem(x)
        geo_feat = geo
        g1 = self.film1.gamma_net(geo_feat); b1 = self.film1.beta_net(geo_feat)
        x = self.film1(x, g1, b1)
        x = self.down1(x)
        g2 = self.film2.gamma_net(geo_feat); b2 = self.film2.beta_net(geo_feat)
        x = self.film2(x, g2, b2)
        x = self.down2(x)
        g3 = self.film3.gamma_net(geo_feat); b3 = self.film3.beta_net(geo_feat)
        x = self.film3(x, g3, b3)
        x = self.down3(x)
        g4 = self.film4.gamma_net(geo_feat); b4 = self.film4.beta_net(geo_feat)
        x = self.film4(x, g4, b4)
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
        self.stem = nn.Sequential(ConvBlock(3, 64, 2), nn.MaxPool2d(3,2,1))
        self.layer1 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128))
        self.layer2 = nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256))
        self.layer3 = nn.Sequential(ResBlock(256, 512, 2), ResBlock(512, 512))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.country_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, NUM_CLASSES))
        self.coord_head = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 2))
    def forward(self, x, coords=None):
        x = self.stem(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        feat = torch.flatten(self.pool(x), 1)
        return self.country_head(feat), self.coord_head(feat)

@register("17_MultiTask-Geo")
def model_17():
    return MultiTaskGeo()

# ============================================================
# MODEL 18: EfficientNet-Style (~4M)
# ============================================================
class MBConvBlock(nn.Module):
    def __init__(self, in_c, out_c, expand_ratio=4, stride=1, se_ratio=4):
        super().__init__()
        mid = in_c * expand_ratio
        self.use_res = stride == 1 and in_c == out_c
        self.expand = nn.Sequential(conv1x1(in_c, mid), nn.BatchNorm2d(mid), nn.SiLU()) if expand_ratio > 1 else nn.Identity()
        self.dw = nn.Sequential(
            nn.Conv2d(mid if expand_ratio>1 else in_c, mid if expand_ratio>1 else in_c, 3, stride, 1, groups=mid if expand_ratio>1 else in_c, bias=False),
            nn.BatchNorm2d(mid if expand_ratio>1 else in_c), nn.SiLU())
        se_mid = max(1, (mid if expand_ratio>1 else in_c) // se_ratio)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), conv1x1(mid if expand_ratio>1 else in_c, se_mid),
                                nn.SiLU(), conv1x1(se_mid, mid if expand_ratio>1 else in_c), nn.Sigmoid())
        self.project = nn.Sequential(conv1x1(mid if expand_ratio>1 else in_c, out_c), nn.BatchNorm2d(out_c))
    def forward(self, x):
        identity = x
        x = self.expand(x); x = self.dw(x)
        x = x * self.se(x)
        x = self.project(x)
        return x + identity if self.use_res else x

class EfficientStyleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(ConvBlock(3, 32, 2))
        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, expand_ratio=1),
            MBConvBlock(16, 24, expand_ratio=4, stride=2),
            MBConvBlock(24, 24, expand_ratio=4),
            MBConvBlock(24, 40, expand_ratio=4, stride=2),
            MBConvBlock(40, 40, expand_ratio=4),
            MBConvBlock(40, 80, expand_ratio=6, stride=2),
            MBConvBlock(80, 80, expand_ratio=6),
            MBConvBlock(80, 112, expand_ratio=6),
            MBConvBlock(112, 192, expand_ratio=6, stride=2),
            MBConvBlock(192, 192, expand_ratio=6),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(192, NUM_CLASSES))
    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.blocks(self.stem(x))), 1))

@register("18_EfficientNet-Style")
def model_18():
    return EfficientStyleNet()

# ============================================================
# MODEL 19: RepVGG-Style (~3M)
# ============================================================
class RepVGGBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.branch_3x3 = nn.Sequential(conv3x3(in_c, out_c, stride), nn.BatchNorm2d(out_c))
        self.branch_1x1 = nn.Sequential(conv1x1(in_c, out_c, stride), nn.BatchNorm2d(out_c))
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.branch_3x3(x) + self.branch_1x1(x))

class RepVGGStyleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stages = nn.Sequential(
            RepVGGBlock(3, 64, 2), RepVGGBlock(64, 64),
            RepVGGBlock(64, 128, 2), RepVGGBlock(128, 128),
            RepVGGBlock(128, 256, 2), RepVGGBlock(256, 256), RepVGGBlock(256, 256), RepVGGBlock(256, 256),
            RepVGGBlock(256, 512, 2), RepVGGBlock(512, 512),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, NUM_CLASSES)
    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.stages(x)), 1))

@register("19_RepVGG-Style")
def model_19():
    return RepVGGStyleNet()

# ============================================================
# MODEL 20: Compact Ensemble-Ready (~800K)
# ============================================================
class CompactNet(nn.Module):
    """Tiny efficient net for ensembling or quick experiments."""
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
