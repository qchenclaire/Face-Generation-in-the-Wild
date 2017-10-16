from PIL import Image
import random
import matplotlib.image as mpimg
#import matplotlib.pyplot as plt
#import numpy as np
import pdb
import scipy.misc
import cv2
import torch
from torch.autograd import Variable
from torchvision import transforms
import torchvision.utils  as tov
import time
def cv2_tensor(pic):
    img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # shape[0] = h, shape[1] = w
    img = img.view(pic.shape[0], pic.shape[1], 3)
    # put it in CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 2).transpose(1, 2).contiguous()
    return img.float().div(256)
def getlist(listname,shuffle=True):
    fr=open(listname)
    datalist=fr.readlines()
    if shuffle:
        random.shuffle(datalist)
    return datalist
def load_data(datapath,listname,datalist,iternow,batch_size,shuffle=True,gpuid=0):
    image=[]
    label=[]
    #face_mean = np.ndarray(shape = (128,128,3),dtype=np.float)
    #face_mean[:,:,0] = 133
    #face_mean[:,:,1] = 105
    #face_mean[:,:,2] = 95
    if iternow*batch_size+batch_size>=len(datalist):
        print 'restart'
        iternow=0
        datalist=getlist(listname)
    for item in datalist[iternow*batch_size:iternow*batch_size+batch_size]:
        a=item.split()
        impath=datapath+a[0]
        im=cv2.imread(impath)
        #tic=time.clock()
        b,g,r = cv2.split(im)
        im = cv2.merge([r,g,b])
        #toc=time.clock()
        #print toc-tic
        im=cv2.resize(im,(64,64))
        #cv2.imwrite('test1.jpg',im)
        im=cv2_tensor(im)
        #scipy.misc.imsave('outfile.jpg',im)
        image.append(im)
    image=Variable(torch.stack(image)).cuda(gpuid)
    #saveim=image.cpu().data
    #tov.save_image(saveim,'test.jpg')
    iternow+=1
    return image,iternow
'''
bs=8
iternow1=0
datalistpo=getlist('list_attr_train1.txt')
imgpo,iternow1=load_data('/ssd/randomcrop_resize_64/','list_attr_train1.txt',datalistpo,iternow1,bs)
print imgpo
'''
