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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
class decoder_meta(nn.Module):
    def __init__(self):
        super(decoder,self).__init__()
        self.fc1=nn.Linear(128,128*2)
        self.fc2=nn.Linear(128*2,128*4)
            
        self.conv2=nn.Sequential(
            nn.ConvTranspose2d(128*4+3, 128 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128 * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(128 * 2, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. (128) x 32 x 32
            nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
            nn.Sigmoid()
            
            # state size. (nc) x 64 x 64
            )
    def forward(self,mask,code):
        #print code.size()
        x=self.fc1(code)
        x=self.fc2(x)
        #print x.size()
        x=x.view(-1,128*4,1,1)
        #print x.size()
        temp=x.repeat(1,1,8,8)
        #print temp.size()
        #print mask.size()
        #print temp.size()
        temp1=torch.cat([temp,mask],1)
        out=self.conv2(temp1)
        return out
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
            nn.Sigmoid()
            )
    def forward(self,x):
        return self.upconv(x)
class upsample_pixel_shuffle(nn.Module):
    def __init__(self):
        super(upsample_pixel_shuffle,self).__init__()
        self.upconv=nn.Sequential(
            nn.Conv2d(1+128*2, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 64*2*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.PixelShuffle(2),

            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 3*2*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(12),
            nn.LeakyReLU(0.2, inplace=True),

            nn.PixelShuffle(2),
            nn.Sigmoid()
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
        mask=F.sigmoid(self.fc4(mask))
        mask=mask.view(-1,3,16,16)
        #print temp.size()
        #print mask.size()
        #print temp.size()
        temp1=torch.cat([code,mask],1)
        #print temp1.size()
        out=self.deconv(temp1)  #see class upsample_deconv() : conv+deconv (6 layers 16*16->64*64)
        return out,mask


def loss_function(recon_x, x, mu, logvar,mu1,logvar1):
    BCE = F.binary_cross_entropy(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD1= -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= bs*64*64*3
    KLD1/=bs*64*64*3

    return BCE + KLD + KLD1
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
lr_rate=0.001
num_iter=500000
bs=128
logging.basicConfig(filename='log/vaenet_v4.log', level=logging.INFO)
vae=VAE().cuda()
optimizer=optim.Adam(vae.parameters(),lr=lr_rate)
datalist=ld.getlist('list_attr_train1.txt')
iternow1=0
'''
state_dict = torch.load('../vaemodelnet2v4/vae_iter_220000.pth.tar')
vae.load_state_dict(state_dict['VAE'])
vae.eval()
imgpo,iternow1=ld.load_data('/ssd/randomcrop_resize_64/','list_attr_train1.txt',datalist,iternow1,bs)
imgpo_re,mu,logvar,mu1,logvar1,mask0,mask1=vae(imgpo)
#mask=vae.deco.fc4(vae.deco.fc3(vae.reparameterize(mu1,logvar1))).view(-1,3,16,16)
#mask=mask/int(torch.max(mask).data.cpu().numpy())
#saveim=imgpo.cpu().data
#tov.save_image(saveim,'img'+'.jpg')
#saveim=imgpo_re.cpu().data
#tov.save_image(saveim,'img_re'+'.jpg')
eps0 = Vb(mu1.data.new(mu1.size()).normal_())
eps1 = Vb(mu.data.new(mu.size()).normal_())
recon,mask2 = vae.deco(eps0,eps1)
saveim=mask0.cpu().data
tov.save_image(saveim,'mask0'+'.jpg')
saveim=mask1.cpu().data
tov.save_image(saveim,'mask1'+'.jpg')
saveim=mask2.cpu().data
tov.save_image(saveim,'mask2'+'.jpg')
saveim=recon.cpu().data
tov.save_image(saveim,'recon'+'.jpg')

'''

'''
eps0 = Vb(mu.data.new(mu.size()).normal_())
sample=vae.deco(mask,eps0)
saveim=sample.cpu().data
tov.save_image(saveim,'sample0'+'.jpg')
saveim=mask.cpu().data
tov.save_image(saveim,'mask0'+'.jpg')
'''
'''
eps1 = Vb(mu.data.new(mu.size()).normal_())
print eps0-eps1
imgpo,iternow1=ld.load_data('/ssd/randomcrop_resize_64/','list_attr_train1.txt',datalist,iternow1,bs)
imgpo_re,mu,logvar,mask1=vae(imgpo)
sample=vae.deco(mask1,eps1)
saveim=sample.cpu().data
tov.save_image(saveim,'sample1'+'.jpg')
saveim=mask1.cpu().data
tov.save_image(saveim,'mask1'+'.jpg')
print mask1-mask
'''

for iter1 in xrange(num_iter):
    vae.train()
    vae.zero_grad()
    #imgpo,iternow1=ld.load_data('/data1/chenqi/celeba_bak/img_align_celeba/','list_attr_train1.txt',datalist,iternow1,bs)
    imgpo,iternow1=ld.load_data('/ssd/randomcrop_resize_64/','list_attr_train1.txt',datalist,iternow1,bs)
    imgpo_re,mu,logvar,mu1,logvar1,mask0,mask1=vae(imgpo)
    loss1=loss_function(imgpo_re,imgpo,mu,logvar,mu1,logvar1)
    #loss2=torch.sum(torch.abs(mask-mask1))/bs
    #loss=loss1+0.001*loss2
    #lossentropy=entropy_loss(mask)
    loss2= F.binary_cross_entropy(mask0, mask1)
    loss=loss1+loss2
    loss.backward()
    optimizer.step()
    
    if iter1%100==0:
        outinfo=str(iter1)+str(loss1)+' '+str(loss1)
        logging.info(outinfo)
        print outinfo
        print iter1
    if iter1 % 200 == 0:
        vae.eval()
        eps0 = Vb(mu.data.new(mu1.size()).normal_()) #gauss noise bs,256 mask
        eps1 = Vb(mu.data.new(mu.size()).normal_())  #gauss noise bs,128 texture
        eps2 = Vb(mu.data.new(mu.size()).normal_())  #gause noise bs,128 texture
        sample,_=vae.deco(eps0,eps1)    # sample
        #save test img
        saveim=sample.cpu().data
        tov.save_image(saveim,'../vaeimgnet2v4/sample0'+str(iter1)+'.jpg')
        sample,_=vae.deco(eps0,eps2)
        saveim=sample.cpu().data
        tov.save_image(saveim,'../vaeimgnet2v4/sample1'+str(iter1)+'.jpg')
        saveim=imgpo_re.cpu().data
        tov.save_image(saveim,'../vaeimgnet2v4/recon'+str(iter1)+'.jpg')
        saveim=imgpo.cpu().data
        tov.save_image(saveim,'../vaeimgnet2v4/img'+str(iter1)+'.jpg')
        #saveim=mask.cpu().data
        #tov.save_image(saveim,'../vaeimgnet2v0/mask'+str(iter1)+'.jpg')
    if iter1 %10000==0:
        # save model
        save_name = '../vaemodelnet2v4/{}_iter_{}.pth.tar'.format('vae', iter1)
        torch.save({'VAE': vae.state_dict()}, save_name)
        logging.info('save model to {}'.format(save_name))
