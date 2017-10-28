import argparse
from torch.autograd import Variable
from vaegan import *
from data_loader import get_loader
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
import models.dcgan as dcgan
import models.mlp as mlp
import torch.nn.init as init
import torchvision.models
import torch.optim as optim
import logging
import pdb
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str, default='./randomcrop_resize_64/')
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--lr_vae', type=float, default=0.0002, help='vae learning rate, default=0.001')
parser.add_argument('--lr_gan', type=float, default=0.0001, help='gan learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--Diters', type=int, default=1, help='number of D iters per each G iter')
parser.add_argument('--mlp_D', action='store_true', default = False, help='use MLP for D')
opt = parser.parse_args()
print(opt)
if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))
logging.basicConfig(filename = '{0}/train.log'.format(opt.experiment), level=logging.INFO)
opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

dataloader = get_loader(image_path=opt.image_path,image_size=opt.image_size,batch_size=opt.batch_size, num_workers=opt.num_workers)
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)
n_extra_layers = int(opt.n_extra_layers)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

input = torch.FloatTensor(opt.batch_size, 3, opt.image_size, opt.image_size)
one = torch.FloatTensor([1])
mone = one * -1
vae=VAE()

if opt.mlp_D:
    netD = mlp.MLP_D(opt.image_size, nz, nc, ndf, ngpu)
else:
    netD = dcgan.DCGAN_D(opt.image_size, nz, nc, ndf, ngpu, n_extra_layers)
    netD.apply(weights_init)

if opt.cuda:
    netD.cuda()
    vae.cuda()
    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()

optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_gan, betas=(opt.beta1, 0.999))
optimizerDec = optim.Adam(vae.deco.parameters(),lr=opt.lr_vae)
optimizerVae = optim.Adam(vae.parameters(),lr=opt.lr_vae)
optimizer = optim.Adam(vae.parameters(),lr=opt.lr_vae)

checkpoint = torch.load('vae_iter_260000.pth.tar')
vae.load_state_dict(checkpoint['VAE'])

gen_iterations = 0
for epoch in range( opt.niter):
    print ('loading data...')
    data_iter = iter(dataloader)
    print ('finish')
    i = 0
    print (len(dataloader))
    while i < len(dataloader):
        ############################
        # (1) Update D network
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update

        #if gen_iterations < 25 or gen_iterations % 500 == 0:
        #    Diters = 100
        #else:
        Diters = opt.Diters
        j = 0
        while j < Diters and i < len(dataloader):
            j += 1

            # clamp parameters to a cube
            for p in netD.parameters():
                p.data.clamp_(opt.clamp_lower, opt.clamp_upper)
            data = data_iter.next()
            i += 1
            # train with real
            real_cpu = data
            netD.zero_grad()
            batch_size = real_cpu.size(0)
            if opt.cuda:
                real_cpu = real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)
            inputv = Variable(input)
            errD_real = netD(inputv)
            errD_real.backward(one, retain_graph=True)

            # train with fake
            recon, mu, logvar, mu1, logvar1, mask0, mask1 = vae(inputv)
            mask_sample = Variable(mu.data.new(mu1.size()).normal_())  # gauss noise bs,256 mask
            texture_sample = Variable(mu.data.new(mu.size()).normal_())  # gauss noise bs,128 texture
            input_sample, _ = vae.deco(mask_sample, texture_sample)  # sample
            errD_sample = netD(input_sample)
            input_recon = recon
            errD_recon = netD(input_recon)
            errD_fake = 0.8 * errD_sample + 0.2 * errD_recon
            errD_fake.backward(mone,retain_graph=True)
            errD = errD_real - errD_fake
            optimizerD.step()
        ############################
        # (2) Update VAE
        ###########################
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        vae.zero_grad()
        mask_sample = Variable(mu.data.new(mu1.size()).normal_())  # gauss noise bs,256 mask
        texture_sample = Variable(mu.data.new(mu.size()).normal_())
        input_sample, _ = vae.deco(mask_sample, texture_sample)
        loss_image = loss_function(recon, inputv, mu, logvar, mu1, logvar1, opt.batch_size)
        loss_mask = F.mse_loss(mask0, mask1)
        loss = 10 *loss_image + 15 * loss_mask
        loss.backward(retain_graph=True)
        errDec_sample = netD(input_sample)
        errDec_sample.backward(one, retain_graph=True)
        input_recon = recon
        errDec_recon = netD(input_recon)
        errDec_sample.backward(one, retain_graph=True)
        errDec = (errDec_sample + errDec_recon)/2
        optimizerVae.step()
        gen_iterations += 1

        if gen_iterations % 200 == 0:
            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                    errD.data[0], errDec.data[0], errD_real.data[0], errD_fake.data[0]))
            print('[%d/%d][%d/%d][%d] Loss_D_sample: %f Loss_D_recon: %f Loss_Dec_sample: %f Loss_D_recon %f'
                % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                    errD_sample.data[0], errD_recon.data[0], errDec_sample.data[0], errDec_recon.data[0]))
            print('[%d/%d][%d/%d][%d] Loss_vae: %f Loss_image: %f Loss_mask: %f'
                % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                    loss.data[0], loss_image.data[0], loss_mask.data[0]))
            logging.info('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f\n'
                % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                    errD.data[0], errDec.data[0], errD_real.data[0], errD_fake.data[0]))
            logging.info('[%d/%d][%d/%d][%d] Loss_D_sample: %f Loss_D_recon: %f Loss_Dec_sample: %f Loss_D_recon %f\n'
                % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                    errD_sample.data[0], errD_recon.data[0], errDec_sample.data[0], errDec_recon.data[0]))
            logging.info('[%d/%d][%d/%d][%d] Loss_vae: %f Loss_image: %f Loss_mask: %f\n'
                % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                    loss.data[0], loss_image.data[0], loss_mask.data[0]))

        if gen_iterations % 200 == 0:
            vae.eval()
            real_cpu = real_cpu.mul(0.5).add(0.5)
            vutils.save_image(real_cpu, '{0}/real_{1}.png'.format(opt.experiment, gen_iterations))
            input_sample.data = input_sample.data.mul(0.5).add(0.5)
            vutils.save_image(input_sample.data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))
            input_recon.data = input_recon.data.mul(0.5).add(0.5)
            vutils.save_image(input_recon.data, '{0}/recon_{1}.png'.format(opt.experiment, gen_iterations))

        if gen_iterations %10000==0:
            # save model
            save_name = './vae_gan/{}_iter_{}.pth.tar'.format('vae', gen_iterations)
            torch.save({'VAE': vae.state_dict()}, save_name)
            logging.info('save model to {}'.format(save_name))
            save_name = './vae_gan/{}_iter_{}.pth.tar'.format('gan', gen_iterations)
            torch.save({'D': netD.state_dict()}, save_name)
            logging.info('save model to {}'.format(save_name))
