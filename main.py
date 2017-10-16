import torch
from torch.autograd import Variable as Vb
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models
import torch.optim as optim
import load_data as ld
import os
import logging
import torchvision.utils  as tov
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
            # state size. 128 x 16 x 16
            nn.Conv2d(128, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128,1,3,1,1,bias=False),
            nn.Sigmoid()
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

            )
        self.fc1=nn.Linear(4*4*128,128)
        self.fc2=nn.Linear(4*4*128,128)
    def forward(self,x):
        feature=self.main(x)
        outmask= self.mask(feature)
        temp=self.texture(feature)
        mu=self.fc1(temp.view(-1,4*4*128))
        logvar=self.fc2(temp.view(-1,4*4*128))
        return outmask,mu,logvar
class decoder(nn.Module):
    def __init__(self):
        super(decoder,self).__init__()
        self.baseunit=nn.Sequential(
            nn.ConvTranspose2d(128, 128* 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128*8),
            nn.ReLU(True),
            # state size. (128*8) x 4 x 4
            nn.ConvTranspose2d(128 * 8, 128 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 4),
            nn.ReLU(True),
            # state size. (128*4) x 8 x 8
            nn.ConvTranspose2d(128 * 4, 128 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 2),
            nn.ReLU(True)
            # state size. (128*2) x 16 x 16
            )
        self.conv2=nn.Sequential(
            nn.ConvTranspose2d(128 * 2+1, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (128) x 32 x 32
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()
            
            # state size. (nc) x 64 x 64
            )
    def forward(self,mask,code):
        x=code.view(-1,128,1,1)
        temp=self.baseunit(x)
        #print temp.size()
        #print mask.size()
        temp1=torch.cat([temp,mask],1)
        out=self.conv2(temp1)
        return out

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
    # Normalise by same number of elements as in reconstruction
    #KLD /= args.batch_size * 784

    return BCE + KLD
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
        mask,mu,logvar=self.enco(x)
        z=self.reparameterize(mu, logvar)
        x_re=self.deco(mask,z)
        mask1,mu1,logvar1=self.enco(x_re)
        return x_re,mu,logvar,mask,mask1
lr_rate=0.001
num_iter=500000
bs=64
logging.basicConfig(filename='log/vae.log', level=logging.INFO)
vae=VAE().cuda()
optimizer=optim.Adam(vae.parameters(),lr=lr_rate)
datalist=ld.getlist('list_attr_train1.txt')
iternow1=0

for iter1 in xrange(num_iter):
    vae.zero_grad()
    imgpo,iternow1=ld.load_data('/ssd/randomcrop_resize_64/','list_attr_train1.txt',datalist,iternow1,bs)
    imgpo_re,mu,logvar,mask,mask1=vae(imgpo)
    loss1=loss_function(imgpo_re,imgpo,mu,logvar)
    loss2=torch.sum(torch.abs(mask-mask1))/bs
    loss=loss1+0.001*loss2
    loss.backward()
    optimizer.step()
    
    if iter1%100==0:
        outinfo=str(loss1)+' '+str(loss1)
        logging.info(outinfo)
        print outinfo
        print iter1
    if iter1 % 200 == 0:
        saveim=imgpo_re.cpu().data
        tov.save_image(saveim,'img/recon'+str(iter1)+'.jpg')
        saveim=imgpo.cpu().data
        tov.save_image(saveim,'img/img'+str(iter1)+'.jpg')
        save_name = 'model/{}_iter_{}.pth.tar'.format('vae', iter1)
        torch.save({'VAE': vae.state_dict()}, save_name)
        logging.info('save model to {}'.format(save_name))

