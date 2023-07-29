import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np

from ldm.ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class MLP_AC_D(nn.Module):
    def __init__(self, opt): 
        super(MLP_AC_D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.ndh)
        self.disc_linear = nn.Linear(opt.ndh, 1)
        self.aux_linear = nn.Linear(opt.ndh, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x):
        h = self.lrelu(self.fc1(x))
        s = self.sigmoid(self.disc_linear(h))
        a = self.aux_linear(h)
        return s,a 

class MLP_AC_2HL_D(nn.Module):
    def __init__(self, opt): 
        super(MLP_AC_2HL_D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.disc_linear = nn.Linear(opt.ndh, 1)
        self.aux_linear = nn.Linear(opt.ndh, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

        self.apply(weights_init)

    def forward(self, x):
        h = self.dropout(self.lrelu(self.fc1(x)))
        h = self.dropout(self.lrelu(self.fc2(h)))
        s = self.sigmoid(self.disc_linear(h))
        a = self.aux_linear(h)
        return s,a 

class MLP_3HL_CRITIC(nn.Module):
    def __init__(self, opt): 
        super(MLP_3HL_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc3 = nn.Linear(opt.ndh, opt.ndh)
        self.fc4 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        h = self.lrelu(self.fc1(h))
        h = self.lrelu(self.fc2(h))
        h = self.lrelu(self.fc3(h))
        h = self.fc4(h)
        return h

class MLP_2HL_CRITIC(nn.Module):
    def __init__(self, opt): 
        super(MLP_2HL_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc3 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        h = self.lrelu(self.fc1(h))
        h = self.lrelu(self.fc2(h))
        h = self.fc3(h)
        return h

class MLP_2HL_Dropout_CRITIC(nn.Module):
    def __init__(self, opt): 
        super(MLP_2HL_Dropout_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc3 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.dropout = nn.Dropout(p=0.5)
        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        h = self.dropout(self.lrelu(self.fc1(h)))
        h = self.dropout(self.lrelu(self.fc2(h)))
        h = self.fc3(h)
        return h

class MLP_CRITIC(nn.Module):
    def __init__(self, opt):
        super(MLP_CRITIC, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h

class MLP_D(nn.Module):
    def __init__(self, opt):
        super(MLP_D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize * 2, opt.ndh)
        #self.fc2 = nn.Linear(opt.ndh, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.dropout = nn.Dropout(p=0.5)
        self.apply(weights_init)

    def forward(self, x, att, s):
        h = torch.cat((x, att, s), 1)
        h = self.fc1(h)
        h = self.lrelu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return h

class MLP_2HL_Dropout_G(nn.Module):
    def __init__(self, opt):
        super(MLP_2HL_Dropout_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.dropout(self.lrelu(self.fc1(h)))
        h = self.dropout(self.lrelu(self.fc2(h)))
        h = self.relu(self.fc3(h))
        return h

class MLP_3HL_G(nn.Module):
    def __init__(self, opt):
        super(MLP_3HL_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.ngh)
        self.fc4 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.lrelu(self.fc2(h))
        h = self.lrelu(self.fc3(h))
        h = self.relu(self.fc4(h))
        return h

class MLP_2HL_G(nn.Module):
    def __init__(self, opt):
        super(MLP_2HL_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.lrelu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return h

class MLP_Dropout_G(nn.Module):
    def __init__(self, opt):
        super(MLP_Dropout_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.2)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.dropout(self.lrelu(self.fc1(h)))
        h = self.relu(self.fc2(h))
        return h

class MLP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    # def forward(self, noise, att, s):
    def forward(self, noise, att):
        # uneven batch size
        noise = noise[:att.shape[0], :]
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h

class MLP_2048_1024_Dropout_G(nn.Module):
    def __init__(self, opt):
        super(MLP_2048_1024_Dropout_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        #self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, 1024)
        self.fc3 = nn.Linear(1024, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        #self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(p=0.5)
        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.dropout(self.lrelu(self.fc1(h)))
        h = self.dropout(self.lrelu(self.fc2(h)))
        h = self.fc3(h)
        return h

class MLP_SKIP_G(nn.Module):
    def __init__(self, opt):
        super(MLP_SKIP_G, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.fc_skip = nn.Linear(opt.attSize + opt.nz, opt.resSize)

        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)
        
        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        #h = self.lrelu(self.fc2(h))
        h = self.relu(self.fc2(h))
        h2 = self.fc_skip(att)
        return h+h2

class MLP_SKIP_D(nn.Module):
    def __init__(self, opt): 
        super(MLP_SKIP_D, self).__init__()
        self.fc1 = nn.Linear(opt.resSize + opt.attSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, 1)
        self.fc_skip = nn.Linear(opt.attSize, opt.ndh)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1) 
        h = self.lrelu(self.fc1(h))
        h2 = self.lrelu(self.fc_skip(att))
        h = self.sigmoid(self.fc2(h+h2))
        return h


class MLP_DIF_G_OLD(nn.Module):
    def __init__(self, opt):
        super(MLP_DIF_G_OLD, self).__init__()

        time_embed_dim = opt.nz * 4
        self.time_embed = nn.Sequential(
            linear(opt.ngh, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.encoder = nn.Sequential(
            nn.Linear(time_embed_dim + opt.nz, opt.ngh),
            nn.LeakyReLU(0.2, True),
            nn.Linear(opt.ngh, opt.ngh),
            nn.LeakyReLU(0.2, True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(opt.ngh + opt.attSize, opt.ngh),
            nn.LeakyReLU(0.2, True),
            nn.Linear(opt.ngh, opt.resSize),
            nn.ReLU(),
        )

        self.model_channels = opt.ngh

        self.apply(weights_init)

    def forward(self, noise, timesteps, context):

        noise = noise.squeeze()
        context = context.squeeze()

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        h = torch.cat((noise, emb), 1)
        h = self.encoder(h)
        h2 = torch.cat((h, context), 1)
        h2 = self.decoder(h2)

        return h2

class MLP_DIF_FUSION(nn.Module):
    def __init__(self, opt):
        super(MLP_DIF_FUSION, self).__init__()

        self.AEmodel = MLP_DIF_G(opt)

        self.FSmodel = AdainResBlk_FUSION(opt)

        self.apply(weights_init)

    def forward(self, xh, timesteps, context):

        cuisines, ingredients = context

        cui_gfeatures = self.AEmodel(xh, timesteps, cuisines)
        ing_gfeatures = self.AEmodel(xh, timesteps, ingredients)

        fused_features = self.FSmodel(cui_gfeatures, ing_gfeatures)

        return fused_features

class AdainResBlk_FUSION(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.resSize)
        self.fc2 = nn.Linear(opt.resSize, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.norm1 = AdaIN1(opt.resSize, opt.resSize)
        self.norm2 = AdaIN1(opt.resSize, opt.resSize)

        self.apply(weights_init)

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.lrelu(x)
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.norm2(x, s)
        x = self.lrelu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        return out

class AdaIN1(nn.Module):

    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)

        gamma, beta = torch.chunk(h, chunks=2, dim=2)

        return ((1 + gamma) * self.norm(x) + beta)

class MLP_DIF_G(nn.Module):
    def __init__(self, opt):
        super(MLP_DIF_G, self).__init__()

        self.model_channels = opt.resSize
        time_embed_dim = self.model_channels * 4
        self.time_embed = nn.Sequential(
            linear(self.model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                time_embed_dim,
                opt.resSize,
            ),
        )

        self.encoder = nn.Sequential(
            # nn.Linear(opt.resSize + opt.attSize, opt.ngh * 2),
            nn.Linear(opt.resSize + opt.resSize + opt.attSize, opt.ngh * 2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(opt.ngh * 2, opt.ngh),
            nn.LeakyReLU(0.2, True),
        )

        self.decoder = nn.Sequential(
            # nn.Linear(opt.ngh + opt.attSize, opt.ngh * 2),
            nn.Linear(opt.ngh + opt.resSize + opt.attSize, opt.ngh * 2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(opt.ngh * 2, opt.resSize),
            nn.ReLU(),
        )

        self.apply(weights_init)

    def forward(self, xh, timesteps, context):
        xh = xh.squeeze()
        context = context.squeeze()

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels, repeat_only=False))
        emb = self.emb_layers(emb)

        h = xh + emb

        h1 = torch.cat((h, context), 1)
        h1 = self.encoder(h1)
        h2 = torch.cat((h1, context), 1)
        h2 = self.decoder(h2)

        h2 = h2.unsqueeze(-2)

        return h2

class ResBlk(nn.Module):

    def __init__(self, opt):
        super(ResBlk, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.apply(weights_init)

    def forward(self, noise, att):
        # uneven batch size
        noise = noise[:att.shape[0], :]
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.lrelu(self.fc2(h))
        return h

class AdaIN(nn.Module):

    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)

        h = h.view(h.size(0), 1, h.size(1))

        gamma, beta = torch.chunk(h, chunks=2, dim=2)

        x = x.view(x.shape[0], 1, x.shape[1])

        return ((1 + gamma) * self.norm(x) + beta).squeeze()


class AdainResBlk(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.fc1 = nn.Linear(opt.ngh, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)

        self.norm1 = AdaIN(opt.attSize, opt.ngh)
        self.norm2 = AdaIN(opt.attSize, opt.ngh)

        self.apply(weights_init)

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.lrelu(x)
        x = self.fc1(x)
        x = self.lrelu(x)
        x = self.norm2(x, s)
        x = self.lrelu(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        return out

class Generator(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.encode = ResBlk(opt)
        self.decode = AdainResBlk(opt)  # stack-like

    def forward(self, noise, x, s):

        x = self.encode(noise, x)
        x = self.decode(x, s)
        return x
