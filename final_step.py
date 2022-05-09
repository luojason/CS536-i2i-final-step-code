#!/usr/bin/env python
# coding: utf-8

# # CS536 Image-to-Image Translation Final Step

# In[1]:


from os import path
from typing import Dict

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision import transforms as tforms
from torchvision import utils as vutils


# In[2]:


torch.backends.cudnn.benchmark = True

ngpu = 1  # code cannot support multi-gpu at this time
device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
print(f'Using {device} device')


# # Architecture Manipulation Code

# In[3]:


def set_device(model: nn.Module, device, ngpu):
    """Transfers the models onto the specified device.
        
    Params:
        model -- which model to transfer
        device -- which device to use
        ngpus -- how many gpus to use, if using cuda device
    Returns:
        The transferred model
    """
    model.to(device)
    if device.type == 'cuda' and ngpu > 1:
        model = nn.DataParallel(model, list(range(ngpu)))
    return model


# In[4]:


def save_network(model: nn.Module, dir_name, name):
    """Saves the current state dictionary to a files in given path.
    
    Params:
        model -- the model to save
        dir_name -- directory in which to store the saved files
        name -- name that is prefixed on the files
    """
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), path.join(dir_name, f'{name}.pth'))
    else:
        torch.save(model.state_dict(), path.join(dir_name, f'{name}.pth'))


# In[5]:


def load_network(model: nn.Module, dir_name, name):
    """Loads a state dictionary from the files in given path.
    
    Params:
        model -- the model to load
        dir_name -- directory from which to load the state
        name -- name that is prefixed on the files
    """
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(path.join(dir_name, f'{name}.pth')))
    else:
        model.load_state_dict(torch.load(path.join(dir_name, f'{name}.pth')))


# # Architecture Definition
# 
# The following define the basic building blocks of our architectures.

# In[6]:


def conv_norm_relu(n_in, n_out, kernel_size, stride, padding=0, padding_mode='reflect', transpose=False, **kwargs):
    """Standard convolution -> instance norm -> relu block.

    Params:
        n_in -- number of input channels
        n_out -- number of filters/output channels
        kernel_size -- passed to Conv2d
        stride -- passed to Conv2d
        padding -- passed to Conv2d
        padding_mode -- passed to Conv2d
        transpose -- whether to use a regular or transposed convolution layer
        kwargs -- other args passed to Conv2d
    Returns:
        A list containing a convolution, instance norm, and ReLU activation
    """
    if transpose:
        conv = nn.ConvTranspose2d(n_in, n_out, kernel_size, stride, padding, padding_mode=padding_mode, bias=True, **kwargs)
    else:
        conv = nn.Conv2d(n_in, n_out, kernel_size, stride, padding, padding_mode=padding_mode, bias=True, **kwargs)
    return [conv, nn.InstanceNorm2d(n_out), nn.ReLU(True)]


# In[7]:


def conv_norm_leakyrelu(n_in, n_out, slope=0.2, **kwargs):
    """Standard convolution -> instance norm -> leaky relu block.
    
    Params:
        n_in -- number of input channels
        n_out -- number of filters/output channels
        slope -- slope of the leaky ReLU layer
        kwargs -- other args passed to the convolution layer
    Returns:
        A list containing a convolution, instance norm, and LeakyReLU activation
    """
    conv = nn.Conv2d(n_in, n_out, **kwargs)
    return [conv, nn.InstanceNorm2d(n_out), nn.LeakyReLU(slope, True)]


# In[8]:


class ResidualBlock(nn.Module):
    """Defines a residual block with 2 3x3 conv-norm-relu layers."""
    
    def __init__(self, k, p=None):
        """Initialize a residual block.
        
        Params:
            k -- number of input and output channels
            p -- dropout rate (optional)
        """
        super().__init__()
        model = conv_norm_relu(k, k, 3, 1, 1)
        model.append(nn.Conv2d(k, k, 3, 1, 1, padding_mode='reflect', bias=True))
        model.append(nn.InstanceNorm2d(k))
        if p is not None:
            model.append(nn.Dropout(p, inplace=True))
        self.block = nn.Sequential(*model)
    
    def forward(self, input):
        residual = self.block(input)
        residual += input  # apply skip-connection
        return residual


# ## Generators
# 
# The generators are VAEs.
# An input image is encoded into a MVN distribution on the latent space with identity covariance,
# whereas the decoder is deterministic (unless dropout is enabled).
# 
# To perform translation, different encoders are paired with different decoders;
# this assumes a shared latent space representation.

# In[9]:


class Encoder(nn.Module):
    """Convolutional-Resnet style encoder."""

    def __init__(self, n_head, n_res, in_channel, n_filter):
        """Initialize an encoder.
        
        Params:
            n_head -- number of downsampling convolution blocks at the head
            n_res -- number of residual blocks in the middle
            in_channel -- number of channels in the input
            n_filter -- number of filters to start with; doubles for each block in the head
        """
        super().__init__()
        # initial convolution
        front = conv_norm_relu(in_channel, n_filter, 7, 1, 3)
        # downsampling convolution blocks
        for _ in range(n_head):
            front += conv_norm_relu(n_filter, 2 * n_filter, 4, 2, 1)
            n_filter *= 2
        # middle residual blocks
        front += [ResidualBlock(n_filter) for _ in range(n_res)]
        self.model = nn.Sequential(*front)
        self.out_channel = n_filter  # record the number of filters before the adjustment
    
    def forward(self, input):
        return self.model(input)


# In[10]:


class LatentAE(nn.Module):
    """Shared latent space VAE. Contains both encoder and decoder."""

    def __init__(self, n_res, in_channel, p=None):
        """Initialize a VAE.

        Params:
            n_res -- number of residual blocks for both the encoder and decoder
            n_channels -- number of channels in the input
            p -- dropout probability used (optional)
        """
        super().__init__()
        self.enc = nn.Sequential(*[ResidualBlock(in_channel) for _ in range(n_res)])
        self.dec = nn.Sequential(*[ResidualBlock(in_channel, p) for _ in range(n_res)])
        self.is_dist = False
    
    def forward(self, input):
        latent_mean = self.enc(input)
        sample = latent_mean + torch.randn(latent_mean.size(), device=latent_mean.device)
        return latent_mean, self.dec(sample)
    


# In[11]:


class Decoder(nn.Module):
    """Convolutional-Resnet style decoder."""

    def __init__(self, n_tail, n_res, in_channel, out_channel, p=None):
        """Initialize a decoder.

        Params:
            n_tail -- number of upsampling convolution blocks at the tail
            n_res -- number of residual blocks in the middle
            in_channel -- number of channels in the input
            out_channel -- desired number of channels in the output
            p -- dropout probability used (optional)
        """
        super().__init__()
        # residual blocks in the middle
        model = [ResidualBlock(in_channel, p) for _ in range(n_res)]
        # upsampling transposed convolution blocks
        for _ in range(n_tail):
            model += conv_norm_relu(in_channel, in_channel // 2, 4, 2, 1, padding_mode='zeros', transpose=True)
            in_channel //= 2
        # final convolution (use tanh)
        model += [nn.Conv2d(in_channel, out_channel, 7, 1, 3, padding_mode='reflect', bias=True), nn.Tanh()]
        self.model = nn.Sequential(*model)
    
    def forward(self, input):
        return self.model(input)


# In[12]:


class Translator(nn.Module):
    """Wraps the models necessary to perform translation between two domains."""

    def __init__(self, d1: str, d2: str, n_channel, n_conv, n_res, n_shared, n_filter, p=None):
        """Initializes two VAEs with shared inner weights.

        Params:
            d1 -- name of first domain
            d2 -- name of second domain
            n_channel -- number of input channels of an image
            n_conv -- number of outermost conv/conv-tranpose blocks in the VAE
            n_res -- number of residual blocks in the middle layers of the VAE
            n_shared -- number of residual blocks that are shared
            n_filter -- number of filters to start with in the encoder
            p -- dropout probability in the decoders (optional)
        """
        super().__init__()
        d1_encoder = Encoder(n_conv, n_res, n_channel, n_filter)
        d2_encoder = Encoder(n_conv, n_res, n_channel, n_filter)
        d1_decoder = Decoder(n_conv, n_res, d1_encoder.out_channel, n_channel, p)
        d2_decoder = Decoder(n_conv, n_res, d2_encoder.out_channel, n_channel, p)
        self.encoders = nn.ModuleDict({d1: d1_encoder, d2: d2_encoder})
        self.decoders = nn.ModuleDict({d1: d1_decoder, d2: d2_decoder})
        self.shared = LatentAE(n_shared, d1_encoder.out_channel, p)

    def translate(self, input, source: str, target: str, keep_mean=True, requires_grad=True):
        """Translates a batch of images from the source domain to the target domain.

        Params:
            input -- input image (batch)
            source -- source domain
            target -- target domain (of translation)
            keep_mean -- whether to also return the latent space mean
            requires_grad -- whether to track the computation graph
        Returns:
            The translated image
        """
        if requires_grad:
            l_mean, encoding = self.shared(self.encoders[source](input))
            output = self.decoders[target](encoding)
        else:
            with torch.no_grad():
                l_mean, encoding = self.shared(self.encoders[source](input))
                output = self.decoders[target](encoding)
        if keep_mean:
            return l_mean, output
        else:
            return output


# ## Discriminators
# 
# The discriminators are multi-scale patchGAN discriminators, as used in ACLGAN, MUNIT etc.

# In[13]:


def make_patchdisc(n_layer, in_channel, n_filter):
    """Makes a basic patchGAN discriminator.

    Params:
        n_layer -- number of convolution-block layers
        in_channel -- number of channels in
        n_filter -- number of filters to start with (doubles at each layer)
    """
    # first layer has no instance norm
    model = [nn.Conv2d(in_channel, n_filter, 4, 2, 1, padding_mode='reflect', bias=True), nn.LeakyReLU(0.2, True)]
    for _ in range(n_layer - 1):
        model += conv_norm_leakyrelu(n_filter, n_filter * 2, kernel_size=4, stride=2, padding=1, padding_mode='reflect', bias=True)
        n_filter *= 2
    model.append(nn.Conv2d(n_filter, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'))
    model.append(nn.Sigmoid())  # use sigmoid with MSE loss
    return nn.Sequential(*model)


# In[14]:


class MSDisc(nn.Module):
    """Multi-scale Discriminator."""

    def __init__(self, n_scale, n_layer, in_channel, n_filter):
        """Initialize a discriminator.

        Params:
            n_scale -- number of scales to run discriminators on
            n_layer -- passed to patchGAN
            in_channel -- passed to patchGAN
            n_filter -- passed to patchGAN
        """
        super().__init__()
        self.models = nn.ModuleList([make_patchdisc(n_layer, in_channel, n_filter) for _ in range(n_scale)])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, input):
        outputs = []
        for model in self.models:
            outputs.append(model(input))
            input = self.downsample(input)
        return outputs


# In[15]:


class DomainDiscs(nn.Module):
    """Wraps the discriminators associated with a particular domain."""

    def __init__(self, name: str, n_channel, n_scale, n_layer, n_filter):
        """Initializes two discriminators for the adv and acl losses.
        
        Params:
            name -- name of the corresponding domain
            n_channel -- number of input channels of an image
            n_scale -- number of scales to run discriminators on
            n_layer -- passed to patchGAN (discriminator initialization)
            n_filter -- passed to patchGAN (discriminator initialization)
        """
        super().__init__()
        self.domain = name
        self.adv_disc = MSDisc(n_scale, n_layer, n_channel, n_filter)
        self.acl_disc = MSDisc(n_scale, n_layer, n_channel * 2, n_filter)  # acl-disc takes two images stacked as input


# # Loss Functions

# In[16]:


# vae regularization term
def vae_kl(latent_mean):
    """Computes the KL-divergence of a latent distribution for a VAE.
    
    Params:
        latent_mean -- mean of latent representation
    Returns:
        KL-divergence
    """
    return 0.5 * torch.mean(torch.square(latent_mean))


# In[17]:


_recon_loss = nn.L1Loss()

# generator/vae loss function
def reconstruction_loss(input, latent_mean, output, hp):
    """Computes the reconstruction loss for a VAE.
    
    Params:
        input -- original image
        latent_mean -- mean of latent representation
        output -- reconstructed image
        hp -- dictionary of hyperparameters
    Returns:
        Reconstruction loss
    """
    # likelihood term (L1 loss)
    loss = hp['nll_w'] * _recon_loss(input, output)
    # regularization term (KL-divergence)
    loss += hp['kl_w'] * vae_kl(latent_mean)
    return loss


# In[18]:


_disc_loss = nn.MSELoss()

# multi-scale discriminator evaluation and loss function
def disc_loss(input, disc: MSDisc, label):
    """Evaluates the discriminator on the input and computes the BCE-loss.

    Params:
        input -- input image
        disc -- discriminator to run the image on
        label -- the 'ground truth' label for that image
    Returns:
        Discriminator loss
    """
    loss = 0
    for output in disc(input):
        truth = torch.full(output.size(), fill_value=label, device=output.device)
        loss += _disc_loss(output, truth)
    return loss


# # Model Update Code
# 
# These functions merely accumulate gradients, and do not perform optimization.
# The optimizer step as well as the clearing of gradients needs to be done outside these functions.

# In[19]:


# labels used in the loss functions
real_label = 1.0
fake_label = 0.0


# In[20]:


def disc_ae_grad(input, ae: Translator, disc: DomainDiscs, hp):
    """Computes the gradients associated with the real input examples for the discriminators.
    
    Params:
        input -- input image batch
        ae -- auto-encoder used
        disc -- discriminators corresponding to the domain
        hp -- dictionary of hyperparameters
    Returns:
        Computed loss values
    """
    domain = disc.domain
    output = ae.translate(input, domain, domain, keep_mean=False, requires_grad=False)
    adv_loss = hp['adv_w'] * disc_loss(input, disc.adv_disc, real_label)  # pass real example to adv
    adv_loss.backward()
    acl_loss = hp['acl_w'] * disc_loss(torch.cat((input, output), dim=1), disc.acl_disc, real_label)  # pass reconstructed pair to acl
    acl_loss.backward()
    return adv_loss.item(), acl_loss.item()


# In[21]:


def gen_ae_grad(input, ae: Translator, disc: DomainDiscs, hp):
    """Computes the gradients associated with the reconstructed images for the generators.
    
    Params:
        input -- input image batch
        ae -- auto-encoder used
        disc -- discriminators corresponding to the domain
        hp -- dictionary of hyperparameters
    Returns:
        Computed loss values
    """
    domain = disc.domain
    l_mean, output = ae.translate(input, domain, domain)
    r_loss = reconstruction_loss(input, l_mean, output, hp)  # vae reconstruction error
    acl_loss = hp['acl_w'] * disc_loss(torch.cat((input, output), dim=1), disc.acl_disc, fake_label)  # pass reconstructed pair to acl
    total_loss = r_loss + acl_loss
    total_loss.backward()
    return r_loss.item(), acl_loss.item()


# In[22]:


def disc_cycle_grad(input, gen: Translator, source_disc: DomainDiscs, target_disc: DomainDiscs, hp):
    """Computes the discriminator gradients for a cycle translation pass.
    
    Params:
        input -- input image batch
        gen -- generator used for translation
        source_disc -- discriminator for source domain
        target_disc -- discriminator for target domain
        hp -- dictionary of hyperparameters
    Returns:
        Computes loss values
    """
    source = source_disc.domain
    target = target_disc.domain
    translated_img = gen.translate(input, source, target, keep_mean=False, requires_grad=False)
    recovered_img = gen.translate(translated_img, target, source, keep_mean=False, requires_grad=False)

    # compute loss for standard discriminators
    adv_loss = disc_loss(translated_img, target_disc.adv_disc, fake_label)
    adv_loss += disc_loss(recovered_img, source_disc.adv_disc, fake_label)
    adv_loss *= hp['adv_w']
    adv_loss.backward()

    # compute loss for acl discriminator
    acl_loss = hp['acl_w'] * disc_loss(torch.cat((input, recovered_img), dim=1), source_disc.acl_disc, fake_label)
    acl_loss.backward()

    return adv_loss.item(), acl_loss.item()


# In[23]:


def gen_cycle_grad(input, gen: Translator, source_disc: DomainDiscs, target_disc: DomainDiscs, hp):
    """Computes the generator gradients for a cycle translation pass.
    
    Params:
        input -- input image batch
        gen -- generator used for translation
        source_disc -- discriminator for source domain
        target_disc -- discriminator for target domain
        hp -- dictionary of hyperparameters
    Returns:
        Computed loss values
    """
    source = source_disc.domain
    target = target_disc.domain
    translated_img = gen.translate(input, source, target, keep_mean=False)
    l_mean, recovered_img = gen.translate(translated_img, target, source)

    # compute regularization losses
    r_loss = vae_kl(l_mean)
    r_loss *= hp['kl_w']

    # compute loss for standard discriminators
    adv_loss = disc_loss(translated_img, target_disc.adv_disc, real_label)
    adv_loss += disc_loss(recovered_img, source_disc.adv_disc, real_label)
    adv_loss *= hp['adv_w']

    # compute loss for acl discriminator
    acl_loss = hp['acl_w'] * disc_loss(torch.cat((input, recovered_img), dim=1), source_disc.acl_disc, real_label)
    
    total_loss = r_loss + adv_loss + acl_loss
    total_loss.backward()
    return adv_loss.item(), acl_loss.item()


# In[24]:


def disc_bigcycle_grad(input, s: str, b: str, t: str, s_gen: Translator, t_gen: Translator, s_disc: DomainDiscs, t_disc: DomainDiscs, hp):
    """Computes the discriminator gradients for a big cycle translation pass (across two domains).
    
    Params:
        input -- input image batch
        s - source domain
        b - bridge domain
        t - target domain
        s_gen -- source to bridge translator
        t_gen -- target to bridge translator
        s_disc -- disciminator for source domain
        t_disc -- discriminator for target domain
    Returns:
        Computed loss values
    """
    bridge_img = s_gen.translate(input, s, b, keep_mean=False, requires_grad=False)
    translated_img = t_gen.translate(bridge_img, b, t, keep_mean=False, requires_grad=False)
    rec_bridge_img = t_gen.translate(translated_img, t, b, keep_mean=False, requires_grad=False)
    recovered_img = s_gen.translate(rec_bridge_img, b, s, keep_mean=False, requires_grad=False)

    # compute loss for standard discriminators
    adv_loss = disc_loss(translated_img, t_disc.adv_disc, fake_label)
    adv_loss += disc_loss(recovered_img, s_disc.adv_disc, fake_label)
    adv_loss *= hp['adv_w']
    adv_loss.backward()

    # compute loss for acl discriminator
    acl_loss = hp['acl_w'] * disc_loss(torch.cat((input, recovered_img), dim=1), s_disc.acl_disc, fake_label)
    acl_loss.backward()

    return adv_loss.item(), acl_loss.item()


# In[25]:


def gen_bigcycle_grad(input, s: str, b: str, t: str, s_gen: Translator, t_gen: Translator, s_disc: DomainDiscs, t_disc: DomainDiscs, hp):
    """Computes the generator gradients for a big cycle translation pass (across two domains).
    
    Params:
        input -- input image batch
        s - source domain
        b - bridge domain
        t - target domain
        s_gen -- source to bridge translator
        t_gen -- target to bridge translator
        s_disc -- disciminator for source domain
        t_disc -- discriminator for target domain
    Returns:
        Computed loss values
    """
    bridge_img = s_gen.translate(input, s, b, keep_mean=False)
    l_mean, translated_img = t_gen.translate(bridge_img, b, t)
    l_mean2, rec_bridge_img = t_gen.translate(translated_img, t, b)
    l_mean3, recovered_img = s_gen.translate(rec_bridge_img, b, s)

    # compute regularization losses
    r_loss = vae_kl(l_mean)
    r_loss += vae_kl(l_mean2)
    r_loss += vae_kl(l_mean3)
    r_loss *= hp['kl_w']

    # compute loss for standard discriminators
    adv_loss = disc_loss(translated_img, t_disc.adv_disc, real_label)
    adv_loss += disc_loss(recovered_img, s_disc.adv_disc, real_label)
    adv_loss *= hp['adv_w']

    # compute loss for acl discriminator
    acl_loss = hp['acl_w'] * disc_loss(torch.cat((input, recovered_img), dim=1), s_disc.acl_disc, real_label)
    
    total_loss = r_loss + adv_loss + acl_loss
    total_loss.backward()
    return adv_loss.item(), acl_loss.item()


# # Dataloader/Preprocessing
# 
# The dataloader uses the basic `ImageFolder` class,
# meaning the provided directory path should contain a single subdirectory, which contains all the images desired.

# In[26]:


img_size = 256
uncropped_img_size = 268


# In[27]:


def load_train_data(path: str, batch_size=16, num_workers=2):
    data_transform = tforms.Compose([
        tforms.Resize(uncropped_img_size),
        tforms.RandomCrop(img_size),
        tforms.RandomHorizontalFlip(),
        tforms.ToTensor(),
        tforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(root=path, transform=data_transform)
    return DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers
    )


# In[28]:


# like load_train_data but performs no preprocessing/data augmentation
def load_test_data(path: str, batch_size=16, num_workers=2):
    data_transform = tforms.Compose([
        tforms.Resize(img_size),
        tforms.CenterCrop(img_size),
        tforms.ToTensor(),
        tforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(root=path, transform=data_transform)
    return DataLoader(
        dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers
    )


# # Initializing Networks

# In[29]:


def init_weights_kaiming(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

def init_weights_normal(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)


# # Training Code

# ## Test images during training

# In[30]:


def test_translate_images(s_in, b_in, t_in, s: str, b: str, t: str, s_gen: Translator, t_gen: Translator, dir_name, name):
    """Translate a test batch and save the results.

    Params:
        s_in -- input batch from source domain
        b_in -- input batch from bridge domain
        t_in -- input batch from target domain
        s, b, t -- domain names
        s_gen -- translator between source and bridge
        t_gen -- translator between bridge and target
        vae: the shared latent network
        dir_name: directory to save images to
        name: image file prefix
    """
    # bridge translations
    b2s = s_gen.translate(b_in, b, s, keep_mean=False, requires_grad=False)
    b2s2b = s_gen.translate(b2s, s, b, keep_mean=False, requires_grad=False)
    b2t = t_gen.translate(b_in, b, t, keep_mean=False, requires_grad=False)
    b2t2b = t_gen.translate(b2t, t, b, keep_mean=False, requires_grad=False)
    img = vutils.make_grid(torch.cat((b_in, b2s, b2s2b, b2t, b2t2b)), nrow=b_in.size(0), normalize=True)
    vutils.save_image(img, path.join(dir_name, f'{name}_bridge.png'))

    # source translations
    s2b = s_gen.translate(s_in, s, b, keep_mean=False, requires_grad=False)
    s2b2t = t_gen.translate(s2b, b, t, keep_mean=False, requires_grad=False)
    s2b2t2b = t_gen.translate(s2b2t, t, b, keep_mean=False, requires_grad=False)
    s2b2t2b2s = s_gen.translate(s2b2t2b, b, s, keep_mean=False, requires_grad=False)
    img = vutils.make_grid(torch.cat((s_in, s2b, s2b2t, s2b2t2b2s)), nrow=s_in.size(0), normalize=True)
    vutils.save_image(img, path.join(dir_name, f'{name}_source.png'))

    # target translations
    t2b = t_gen.translate(t_in, t, b, keep_mean=False, requires_grad=False)
    t2b2s = s_gen.translate(t2b, b, s, keep_mean=False, requires_grad=False)
    t2b2s2b = s_gen.translate(t2b2s, s, b, keep_mean=False, requires_grad=False)
    t2b2s2b2t = t_gen.translate(t2b2s2b, b, t, keep_mean=False, requires_grad=False)
    img = vutils.make_grid(torch.cat((t_in, t2b, t2b2s, t2b2s2b2t)), nrow=t_in.size(0), normalize=True)
    vutils.save_image(img, path.join(dir_name, f'{name}_target.png'))


# ## Setting execution parameters

# In[31]:


# architecture parameters (see DomainModels constructor for details)
n_channel = 3
n_conv = 3
n_res = 3
n_shared = 1  # number of shared layers, in both encoder and decoder
n_gen_filter = 64
n_scale = 1  # number of scales to apply MS-disc at
n_layer = 5  # number of layers for the discriminator
n_disc_filter = 64
p = 0.25


# In[32]:


# loss function hyperparameters
hp = {}

# for VAE-loss
hp['nll_w'] = 100
hp['kl_w'] = 0.1

# for discriminator losses
hp['adv_w'] = 100
hp['acl_w'] = 50


# In[33]:


# optimizer hyperparameters (for ADAM optimizer)
lr = 0.0001
beta1 = 0.5
beta2 = 0.999


# In[34]:


# training hyperparameters
disc_update_freq = 1
gen_update_freq = 1
batch_size = 4
test_batch_size = 16  # number of test images to translate and log after each epoch during training
n_epochs = 20  # number of epochs to train for the second phase
checkpoint_freq = 2  # save copy of models every nth epoch
test_freq = 1  # save test translated images every nth epoch


# ## Setting up models, data, and checkpoints
# 
# Domain `A` is the prerecorded synthetic pizza,
# domain `B` is the live synthetic pizza (serving as a bridge),
# and domain `C` is the real pizza.

# In[35]:


# directory paths
checkpoint_dir = 'checkpoints'
result_dir = 'training_results'

A_train_dir = 'prerec_train'
B_train_dir = 'live_train'
C_train_dir = 'real_train'

A_test_dir = 'prerec_test'
B_test_dir = 'live_test'
C_test_dir = 'real_test'


# In[36]:


# setup translators between domains
AB_gen = Translator('A', 'B', n_channel, n_conv, n_res, n_shared, n_gen_filter, p)
BC_gen = Translator('B', 'C', n_channel, n_conv, n_res, n_shared, n_gen_filter, p)

AB_gen.apply(init_weights_kaiming)
BC_gen.apply(init_weights_kaiming)
AB_gen = set_device(AB_gen, device, ngpu)
BC_gen = set_device(BC_gen, device, ngpu)

# setup disciminators for each domain
A_disc = DomainDiscs('A', n_channel, n_scale, n_layer, n_disc_filter)
B_disc = DomainDiscs('B', n_channel, n_scale, n_layer, n_disc_filter)
C_disc = DomainDiscs('C', n_channel, n_scale, n_layer, n_disc_filter)

A_disc.apply(init_weights_normal)
B_disc.apply(init_weights_normal)
C_disc.apply(init_weights_normal)
A_disc = set_device(A_disc, device, ngpu)
B_disc = set_device(B_disc, device, ngpu)
C_disc = set_device(C_disc, device, ngpu)


# In[37]:


# setup optimizers
AB_g_opt = optim.Adam(AB_gen.parameters(), lr=lr, betas=(beta1, beta2))
BC_g_opt = optim.Adam(BC_gen.parameters(), lr=lr, betas=(beta1, beta2))
g_opts = [AB_g_opt, BC_g_opt]

A_d_opt = optim.Adam(A_disc.parameters(), lr=lr, betas=(beta1, beta2))
B_d_opt = optim.Adam(B_disc.parameters(), lr=lr, betas=(beta1, beta2))
C_d_opt = optim.Adam(C_disc.parameters(), lr=lr, betas=(beta1, beta2))
d_opts = [A_d_opt, B_d_opt, C_d_opt]


# In[38]:


# setup dataloaders
A_train = load_train_data(A_train_dir, batch_size)
B_train = load_train_data(B_train_dir, batch_size)
C_train = load_train_data(C_train_dir, batch_size)

A_test = load_test_data(A_test_dir, test_batch_size)
B_test = load_test_data(B_test_dir, test_batch_size)
C_test = load_test_data(C_test_dir, test_batch_size)


# ## Training

# In[39]:


# train
dadv_loss = 0
dacl_loss = 0
gadv_loss = 0
gacl_loss = 0
r_loss = 0
for epoch in range(n_epochs):
    for it, (A_imgs, B_imgs, C_imgs, A_imgs2, B_imgs2, C_imgs2) in enumerate(zip(A_train, B_train, C_train, A_train, B_train, C_train)):
        A_imgs = A_imgs[0].to(device)
        B_imgs = B_imgs[0].to(device)
        C_imgs = C_imgs[0].to(device)
        A_imgs2 = A_imgs2[0].to(device)
        B_imgs2 = B_imgs2[0].to(device)
        C_imgs2 = C_imgs2[0].to(device)

        # update discriminators if needed
        if it % disc_update_freq == 0:
            for d_opt in d_opts:
                d_opt.zero_grad()

            # extra real examples to mitigate bias in discriminators
            adv1, acl1 = disc_ae_grad(A_imgs, AB_gen, A_disc, hp)
            adv2, acl2 = disc_ae_grad(B_imgs, AB_gen, B_disc, hp)
            adv3, acl3 = disc_ae_grad(B_imgs, BC_gen, B_disc, hp)
            adv4, acl4 = disc_ae_grad(C_imgs, BC_gen, C_disc, hp)
            adv5, acl5 = disc_ae_grad(A_imgs2, AB_gen, A_disc, hp)
            adv6, acl6 = disc_ae_grad(B_imgs2, AB_gen, B_disc, hp)
            adv7, acl7 = disc_ae_grad(B_imgs2, BC_gen, B_disc, hp)
            adv8, acl8 = disc_ae_grad(C_imgs2, BC_gen, C_disc, hp)

            # generate translations
            adv9, acl9 = disc_cycle_grad(A_imgs, AB_gen, A_disc, B_disc, hp)
            adv10, acl10 = disc_cycle_grad(B_imgs, AB_gen, B_disc, A_disc, hp)
            adv11, acl11 = disc_cycle_grad(C_imgs, BC_gen, C_disc, B_disc, hp)
            adv12, acl12 = disc_cycle_grad(B_imgs, BC_gen, B_disc, C_disc, hp)
            adv13, acl13 = disc_bigcycle_grad(A_imgs, 'A', 'B', 'C', AB_gen, BC_gen, A_disc, C_disc, hp)
            adv14, acl14 = disc_bigcycle_grad(C_imgs, 'C', 'B', 'A', BC_gen, AB_gen, C_disc, A_disc, hp)

            for d_opt in d_opts:
                d_opt.step()

            dadv_loss = adv1 + adv2 + adv3 + adv4 + adv5 + adv6 + adv7 + adv8 + adv9 + adv10 + adv11 + adv12 + adv13 + adv14
            dacl_loss = acl1 + acl2 + acl3 + acl4 + acl5 + acl6 + acl7 + acl8 + acl9 + acl10 + acl11 + acl12 + acl13 + acl14
        
        # update generators if needed
        if it % gen_update_freq == 0:
            for g_opt in g_opts:
                g_opt.zero_grad()
            
            # image reconstructions
            r1, acl1 = gen_ae_grad(A_imgs, AB_gen, A_disc, hp)
            r2, acl2 = gen_ae_grad(B_imgs, AB_gen, B_disc, hp)
            r3, acl3 = gen_ae_grad(B_imgs, BC_gen, B_disc, hp)
            r4, acl4 = gen_ae_grad(C_imgs, BC_gen, C_disc, hp)

            # generate translations
            adv5, acl5 = gen_cycle_grad(A_imgs, AB_gen, A_disc, B_disc, hp)
            adv6, acl6 = gen_cycle_grad(B_imgs, AB_gen, B_disc, A_disc, hp)
            adv7, acl7 = gen_cycle_grad(C_imgs, BC_gen, C_disc, B_disc, hp)
            adv8, acl8 = gen_cycle_grad(B_imgs, BC_gen, B_disc, C_disc, hp)
            adv9, acl9 = gen_bigcycle_grad(A_imgs, 'A', 'B', 'C', AB_gen, BC_gen, A_disc, C_disc, hp)
            adv10, acl10 = gen_bigcycle_grad(C_imgs, 'C', 'B', 'A', BC_gen, AB_gen, C_disc, A_disc, hp)

            for g_opt in g_opts:
                g_opt.step()
            gadv_loss = adv5 + adv6 + adv7 + adv8 + adv9 + adv10
            gacl_loss = acl1 + acl2 + acl3 + acl4 + acl5 + acl6 + acl7 + acl8 + acl9 + acl10
            r_loss = r1 + r2 + r3 + r4
        
        print(f'[{it}], adv: {dadv_loss:.4f}, acl: {dacl_loss:.4f}, gadv: {gadv_loss:.4f}, gacl: {gacl_loss:.4f}, r: {r_loss:.4f}', end='\n')
    # print()  # keep printed stats for the end of an epoch

    # checkpoints
    save_network(AB_gen, checkpoint_dir, 'AB_latest')
    save_network(BC_gen, checkpoint_dir, 'BC_latest')
    save_network(A_disc, checkpoint_dir, 'A_disc_latest')
    save_network(B_disc, checkpoint_dir, 'B_disc_latest')
    save_network(C_disc, checkpoint_dir, 'C_disc_latest')
    if (epoch + 1) % checkpoint_freq == 0:
        save_network(AB_gen, checkpoint_dir, f'AB_{epoch}')
        save_network(BC_gen, checkpoint_dir, f'BC_{epoch}')
        save_network(A_disc, checkpoint_dir, f'A_disc_{epoch}')
        save_network(B_disc, checkpoint_dir, f'B_disc_{epoch}')
        save_network(C_disc, checkpoint_dir, f'C_disc_{epoch}')
    
    # test images
    if (epoch + 1) % test_freq == 0:
        s_in = next(iter(A_test))[0].to(device)
        b_in = next(iter(B_test))[0].to(device)
        t_in = next(iter(C_test))[0].to(device)
        test_translate_images(s_in, b_in, t_in, 'A', 'B', 'C', AB_gen, BC_gen, result_dir, str(epoch))


# In[ ]:




