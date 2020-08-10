import torch
from torch import nn
import torchvision
import torch.nn.functional as nnf
import torchvision.transforms as transforms
from PIL import Image
import glob, os

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self,
                 frames_dir='test_vid/train',
                 num_in_frames=4,
                 transforms=transforms.Compose([transforms.ToTensor()]),
                 img_size=256):

        self.img_size = img_size
        self.num_in_frames = num_in_frames
        self.transforms = transforms
        self.r_frames = sorted(glob.glob(os.path.join(frames_dir,'*.png')))
        self.f_frames = sorted(glob.glob(os.path.join(frames_dir,'*.jpg')))

    def __getitem__(self, index):
        imgs = []
        for frame in self.f_frames[index:index+self.num_in_frames]:
            imgs.append(self.transforms(Image.open(frame).resize((self.img_size,self.img_size))))

        prior = torch.cat(imgs)
        goal = self.transforms(Image.open(self.r_frames[index+self.num_in_frames]).resize((self.img_size,self.img_size)))

        return {'X':prior, 'Y':goal}

    def __len__(self):
        return len(self.r_frames)-self.num_in_frames

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss

def run_discriminator(batch, unet, disc, L_adv, in_frames, batch_size):
    x, y = batch['X'], batch['Y']
    if torch.cuda.is_available(): x, y = batch['X'].cuda(), batch['Y'].cuda()

    # Run through RECURRENT U-Net
    for i in range(0,in_frames*3,3):
        y_pred  = unet(x[:,i:i+3, :, :])

    # Resize tensors for less expensive Discriminator training
    y_pred = nnf.interpolate(y_pred, size=(512, 512), mode='bilinear', align_corners=False)
    x = nnf.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
    y = nnf.interpolate(y, size=(512, 512), mode='bilinear', align_corners=False)

    ### Adversarial labels
    fake_set = torch.cat([x, y_pred], axis = 1)
    fake_labels = torch.zeros(batch_size, 1, 1, 1)

    real_set = torch.cat([x,y], axis = 1)
    real_labels = torch.ones(batch_size, 1, 1, 1)

    fake_real_imgs = torch.cat([fake_set, real_set])
    fake_real_labels = torch.cat([fake_labels, real_labels])
    if torch.cuda.is_available(): fake_real_labels = fake_real_labels.cuda()
    ###

    adv_error = L_adv(disc(fake_real_imgs), fake_real_labels)

    return adv_error

def run_generator(batch, unet, disc, L_adv, L_pixel, L_percept, in_frames, batch_size):
    x, y = batch['X'], batch['Y']
    if torch.cuda.is_available(): x, y = batch['X'].cuda(), batch['Y'].cuda()

    for i in range(0,in_frames*3,3):
        y_pred = unet(x[:,i:i+3, :, :])

    y_pred = nnf.interpolate(y_pred, size=(1024, 1024), mode='bilinear', align_corners=False)
    pixel_error = L_pixel(y_pred, y)

    y_pred = nnf.interpolate(y_pred, size=(512, 512), mode='bilinear', align_corners=False)
    x = nnf.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
    y = nnf.interpolate(y, size=(512, 512), mode='bilinear', align_corners=False)

    fake_set = torch.cat([x,y_pred], axis = 1)
    real_labels = torch.ones(batch_size, 1, 1, 1)

    g_adv_error = L_adv(disc(fake_set), real_labels.cuda())

    perceptual_error = L_percept(y_pred.cpu(), y.cpu())

    return pixel_error, g_adv_error, perceptual_error, y_pred
