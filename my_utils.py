import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from vgg import VGG

# Load image functions
loader = transforms.Compose([
    transforms.ToTensor()]) 
def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image


def smooth_image(input_img):
    gaussian_filt = torch.tensor([.1523,.2220,.2514,.2220,.1523],dtype=torch.float32,device=torch.device("cuda")) 
    gaussian_filt = gaussian_filt.view(1,-1)
    gaussian_filt = torch.mm(gaussian_filt.t(), gaussian_filt).view(1,1,5,5)
    imsize0 = input_img.size(2)
    imsize1 = input_img.size(3)
    input_img = input_img.view(3,1,imsize0,imsize1).cuda()
    for k in range(2):
        input_img = torch.nn.functional.conv2d(input_img,gaussian_filt,padding=2)
    x = input_img.view(1,3,imsize0,imsize1)
    return x
    
    
def get_input_noise(imsize0,imsize1):
    # filter noise with a gaussian to initialize synthesis
    gaussian_filt = torch.tensor([.1523,.2220,.2514,.2220,.1523],dtype=torch.float32,device=torch.device("cuda")) 
    gaussian_filt = gaussian_filt.view(1,-1)
    gaussian_filt = torch.mm(gaussian_filt.t(), gaussian_filt).view(1,1,5,5)
    input_img = torch.randn(3,1,imsize0,imsize1,dtype=torch.float32,device=torch.device("cuda"))
    for k in range(5):
        input_img = torch.nn.functional.conv2d(input_img,gaussian_filt,padding=2)
    x = input_img.view(1,3,imsize0,imsize1)
    return x

def get_vgg_net(model_folder,out_keys = ['r11','r21','r31','r41','r51']):
    
    
    vgg_net = VGG(pool='avg',out_keys=out_keys)
    vgg_net.load_state_dict(torch.load(model_folder+'vgg_conv.pth'))
    vgg_net.cuda()
    for param in vgg_net.parameters():
        param.requires_grad = False
    return vgg_net