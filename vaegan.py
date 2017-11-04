import os
import torch
from torch.autograd import Variable as Vb
import torch.nn as nn
import torch.nn.functional as F

LAMBDA = 2 # Gradient penalty lambda hyperparameter
class encoder(nn.Module):
    def __init__(self):
        super(encoder,self).__init__()
        self.main = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 32 x 32
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
            )
        self.mask=nn.Sequential(

            nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128,3,1,1,0,bias=False),
            nn.BatchNorm2d(3),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Tanh()
            )
        self.texture=nn.Sequential(
            nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            torch.nn.AvgPool2d(4)
            )
        self.fc1=nn.Linear(128,128)
        self.fc2=nn.Linear(128,128)
        self.fc3=nn.Linear(256*3,256)
        self.fc4=nn.Linear(256*3,256)
    def forward(self,x):
        feature=self.main(x)
        outmask= self.mask(feature)
        mu1=self.fc3(outmask.view(-1,256*3))
        logvar1=self.fc4(outmask.view(-1,256*3))
        temp=self.texture(feature)
        mu=self.fc1(temp.view(-1,128))
        logvar=self.fc2(temp.view(-1,128))
        return mu,logvar,mu1,logvar1,outmask

class upsample_deconv(nn.Module):
    def __init__(self):
        super(upsample_deconv,self).__init__()
        self.upconv=nn.Sequential(
            nn.Conv2d(3+128*2, 128*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128*2, 128*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128*2, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (128) x 32 x 32
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            )
    def forward(self,x):
        return self.upconv(x)


class decoder(nn.Module):
    def __init__(self):
        super(decoder,self).__init__()
        self.fc1=nn.Linear(128,128*2)
        self.fc2=nn.Linear(128*2,128*2)
        self.fc3=nn.Linear(256,256*3)
        self.fc4=nn.Linear(256*3,256*3)
        self.deconv=upsample_deconv()
    def forward(self,mask,code):
        #print code.size()
        code=F.leaky_relu(self.fc1(code))
        code=F.leaky_relu(self.fc2(code))
        #print x.size()
        code=code.view(-1,128*2,1,1)
        #print x.size()
        code=code.repeat(1,1,16,16)
        mask=F.leaky_relu(self.fc3(mask))
        mask=F.tanh(self.fc4(mask))
        mask=mask.view(-1,3,16,16)
        #print temp.size()
        #print mask.size()
        #print temp.size()
        temp1=torch.cat([code,mask],1)
        #print temp1.size()
        out=self.deconv(temp1)
        return out,mask

def loss_function(recon_x, x, mu, logvar,mu1,logvar1, batch_size):
    MSE = F.mse_loss(recon_x, x)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD1= -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= batch_size*64*64*3
    KLD1/= batch_size*64*64*3


    '''print ('logvar')
    print (logvar.max())
    print('mu')
    print(mu.max())
    '''
    return 5 * MSE + 50 * KLD + 50 * KLD1
def entropy_loss(x):
    x1=torch.squeeze(torch.sum(torch.sum(x,2),3))
    return -torch.sum(torch.mul(x,torch.log(x)))+torch.sum(torch.mul(x1,torch.log(x1)))
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.enco=encoder()
        self.deco=decoder()
    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Vb(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu
    def forward(self,x):
        mu,logvar,mu1,logvar1,mask0=self.enco(x)
        maskcode=self.reparameterize(mu1,logvar1)
        code=self.reparameterize(mu, logvar)
        x_re,mask1=self.deco(maskcode,code)
        #mask1,mu1,logvar1=self.enco(x_re)
        return x_re,mu,logvar,mu1,logvar1,mask0,mask1

class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )
    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1).squeeze(1)
def calc_gradient_penalty(netD, real_data, fake_sample, fake_recon):
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(opt.batch_size, real_data.nelement()/opt.batch_size).contiguous().view(opt.batch_size, 3, 64, 64)
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha)/2 * fake_sample) + ((1 - alpha)/2 * fake_recon)


    interpolates = interpolates.cuda()
    interpolates = Vb(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty