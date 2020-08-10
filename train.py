import torch.nn.functional as nnf
from models import ResNet, U_Net, CNN
import utils import VGGPerceptualLoss, run_generator, run_discriminator, VideoDataset
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def init_model(nfm=32,
              res_blocks=1,
              in_frames=2,
              batch_size=2,
              epoch_to_load=None):

    resnet = ResNet(nfm*2, res_blocks)
    if torch.cuda.is_available(): resnet=resnet.cuda()

    my_unet = U_Net(nfm, resnet, 1, 1)
    discriminator = CNN((in_frames+1)*3, nfm, 512)

    if epoch_to_load != None:
        my_unet = torch.load('unet_epoch_{}'.format(epoch_to_load))
        discriminator = torch.load('D_epoch_{}'.format(epoch_to_load))

    if torch.cuda.is_available(): my_unet, discriminator = my_unet.cuda(), discriminator.cuda()

    Unet_optim = torch.optim.Adam(my_unet.parameters(), lr=0.002)
    D_optim = torch.optim.Adam(discriminator.parameters(), lr=0.002)

    return {'Unet': my_unet, 'Discriminator': discriminator, 'Unet_optimizer': Unet_optim, 'Discriminator_optimizer': D_optim}

def train(my_unet,
          discriminator,
          train_dataloader,
          U_optim,
          D_optim,
          test_dataloader=None,
          epochs=50)

    train_losses = []
    test_losses = []

    L_pixel =  nn.MSELoss(reduction='sum')
    L_adv = nn.BCELoss()
    L_percept = VGGPerceptualLoss()

    RS = transforms.Resize((256, 512))
    TPIL = transforms.ToPILImage()
    prevTransforms = transforms.Compose([TPIL, RS])

    print('Training Started...')
    for epoch in range(epochs):
        for num, batch in enumerate(train_dataloader):
            if batch['X'].shape[0] == batch_size:

                hidden = my_unet._init_hidden(batch_size)
                U_optim.zero_grad()
                D_optim.zero_grad()

                adv_error = run_discriminator(batch, my_unet, discriminator, L_adv, in_frames, batch_size)

                adv_error.backward()
                D_optim.step()

                hidden = my_unet._init_hidden(batch_size)
                U_optim.zero_grad()
                D_optim.zero_grad()

                pixel_error, g_adv_error, perceptual_error, y_pred = run_generator(batch, my_unet, discriminator, L_adv, L_pixel, L_percept, in_frames, batch_size)

                error = pixel_error + g_adv_error
                error.backward()
                U_optim.step()

                print('Epoch {}/{}, Batch {}/{}, D_Adv Loss: {}, G_Adv Loss: {}, Pixel Loss: {}, Perceptual Loss: {}'.format(epoch, epochs, num, len(train_dataloader), adv_error.item(), g_adv_error.item(), pixel_error.item(), perceptual_error.item()))
                train_losses.append(pixel_error)

        if epoch % 10==0:
            if test_dataloader != None
                test_error = 0
                for test_num, test_batch in enumerate(test_dataloader):
                    if batch['X'].shape[0] == batch_size:
                        hidden = my_unet._init_hidden(batch_size)
                        optimizer.zero_grad()
                        D_optim.zero_grad()

                        adv_error = run_discriminator(batch, my_unet, discriminator, L_adv, in_frames, batch_size)

                        hidden = my_unet._init_hidden(batch_size)
                        optimizer.zero_grad()
                        D_optim.zero_grad()

                        pixel_error, g_adv_error, perceptual_error, y_pred = run_generator(batch, my_unet, discriminator, L_adv, L_pixel, L_percept, in_frames, batch_size)

                test_losses.append(pixel_error.item())
                print('Epoch {}/{}, Test Pixel Loss: {}, Test Perceptual Loss: {}, Test D_adv Loss: {}, Test G_adv Loss: {}'.format(epoch, epochs,pixel_error.item(), perceptual_error.item(), adv_error.item(), g_adv_error.item()) )

                plt.figure(figsize=(8,5))
                plt.imshow(prevTransforms(torch.cat([y_pred[0].cpu(), nnf.interpolate(test_batch['Y'], size=(512, 512), mode='bilinear', align_corners=False)[0].cpu()], axis=2)))
                plt.savefig('test_sample_epoch_{}.png'.format(epoch))

            torch.save(my_unet, 'unet_epoch_{}'.format(epoch))
            torch.save(discriminator, 'D_epoch_{}'.format(epoch))

        print('Epoch {}/{}, D_adv Loss: {}, G_adv Loss: {}, Pixel Loss: {}'.format(epoch, epochs, adv_error.item(), g_adv_error.item(), pixel_error.item()))

def main():
    in_frames = 2
    batch_size = 2

    train_dataloader = torch.utils.data.DataLoader(VideoDataset(frames_dir='test_vid/train',img_size=1024,num_in_frames=in_frames,),
                                                  batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(VideoDataset(frames_dir='test_vid/test',img_size=1024,num_in_frames=in_frames,),
                                                  batch_size=batch_size, shuffle=False)

    model_dict = init_model(res_blocks=1, in_frames=in_frames, batch_size=batch_size, epoch_to_load=None)
    train(model_dict['Unet'], model_dict['Discriminator'], train_dataloader, model_dict['Unet_optimizer'], model_dict['Discriminator_optimizer'], test_dataloader=test_dataloader, epochs=50)

if __name__ == '__main__':
    main()
