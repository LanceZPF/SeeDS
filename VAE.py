
import torch
import torch.nn as nn

class VAE(nn.Module):

    def __init__(self, opt):

        super().__init__()

        encoder_layer_sizes = [opt.resSize, opt.ngh*2]
        
        latent_size = opt.ngh
        
        decoder_layer_sizes = [opt.ngh*2, opt.resSize]

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, conditional = True, num_labels = opt.attSize)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, conditional = True, num_labels = opt.attSize)

    def forward_(self, x, c=None):

        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)

        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):

        z = z.squeeze()
        c = c.squeeze()

        recon_x = self.decoder(z, c)
        return recon_x

    def loss_fn(self, recon_x, x, mean, log_var):

        BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + KLD) / x.size(0)

    def forward(self, x, c):

        x = x.squeeze()
        c = c.squeeze()

        recon_x, mean, log_var, z = self.forward_(x, c)
        loss = self.loss_fn(recon_x, x, mean, log_var)

        return loss, [loss]


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += num_labels

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.LeakyReLU(0.2, True))

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.LeakyReLU(0.2, True))
            else:
                self.MLP.add_module(name="sigmoid", module=nn.ReLU())

    def forward(self, z, c):

        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x

