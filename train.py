import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import pickle
from cv2 import PSNR
from torchvision import transforms
import matplotlib.pyplot as plt


def meta_learning(model, dataset, epoch, batch_size, inner_num=3, summary_step=100, outer_lr=3e-5, image_size=(32, 32), image_channel=3, save_path='cifar10_functa.pt'):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12)
    criterion = torch.nn.MSELoss(reduction='sum')
    image_dim = image_size[0] * image_size[1] * image_channel
    outer_params = list(model.parameters())
    for siren_layer in model.siren_layers:
        outer_params += list(siren_layer.parameters())
    outer_optim = torch.optim.Adam(outer_params, lr=outer_lr)
    for e in range(epoch):
        for outer_step, (image, _) in enumerate(loader):
            batch_size = image.shape[0]
            image = transforms.Normalize([0.5] * image_channel, [0.5] * image_channel)(image)  # [B, C, H, W]
            pixels = image.permute(0, 2, 3, 1).reshape(-1, image_channel).to(device)  # [B*H*W, C]
            coords = tuple(torch.linspace(-1, 1, steps=l) for l in image_size)
            coords = torch.stack(torch.meshgrid(*coords), dim=-1).reshape(-1, 2)
            coords = coords.repeat(batch_size, 1).view(batch_size, -1, 2).to(device)  # [B, H*W, C]
            for params in outer_params:
                params.requires_grad = False
            latent_params = torch.zeros((batch_size, *model.latent_dim)).to(device)  # [B, L]
            latent_params.requires_grad = True
            inner_optim = torch.optim.SGD([latent_params], lr=1e-2)
            for inner_step in range(inner_num):
                modulations = model.l2m(latent_params)
                logits = model(coords, modulations)
                loss = criterion(logits, pixels) / image_dim
                inner_optim.zero_grad()
                loss.backward()
                inner_optim.step()
            for params in outer_params:
                params.requires_grad = True
            latent_params = latent_params.detach()
            modulations = model.l2m(latent_params)
            logits = model(coords, modulations)
            loss = criterion(logits, pixels) / image_dim
            outer_optim.zero_grad()
            loss.backward()
            outer_optim.step()
            if not outer_step % summary_step:
                print("Epoch {}, outer step{}".format(e, outer_step))
                print("Train loss: {}".format(loss.item()))
                mpsnr = 0
                image_batch = pixels.cpu().view(batch_size, image_size[0], image_size[1], image_channel)
                predicted_batch = logits.cpu().view(batch_size, image_size[0], image_size[1], image_channel)
                for i in range(batch_size):
                    mpsnr += PSNR(256*(image_batch[i].detach().numpy() * 0.5 + 0.5), 256*(predicted_batch[i].detach().numpy() * 0.5 + 0.5))
                print("PSNR {}".format(mpsnr / batch_size))
                fig, axes = plt.subplots(1, 2, figsize=(18, 6))
                axes[0].imshow(image_batch[0].detach().numpy() * 0.5 + 0.5)
                axes[1].imshow(predicted_batch[0].detach().numpy() * 0.5 + 0.5)
                plt.show()
                torch.save(model, save_path)


class MetaSGD:
    def __init__(self, shape, init=1e-2):
        self.alpha = init * torch.ones(shape).to(device)
        self.alpha.requires_grad = True

    def update_weights(self, loss, weights):
        grad = torch.autograd.grad(loss, weights, create_graph=True)
        return weights - torch.mul(self.alpha, grad[0])


def meta_sgd_learning(model, dataset, epoch, batch_size, inner_num=3, summary_step=77, outer_lr=3e-5, image_size=(32, 32), image_channel=3, save_path='cifar10_functa.pt', optim_path='cifar10_optim.pickle', color='gray'):
    meta_sgd = MetaSGD(model.latent_dim)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=12)
    image_dim = image_size[0] * image_size[1] * image_channel
    criterion = torch.nn.MSELoss(reduction='sum')
    outer_params = list(model.parameters())
    for siren_layer in model.siren_layers:
        outer_params += list(siren_layer.parameters())
    outer_optim = torch.optim.Adam(outer_params + [meta_sgd.alpha], lr=outer_lr, weight_decay=1e-4)
    for e in range(epoch):
        for outer_step, (image, _) in enumerate(loader):
            batch_size = image.shape[0]
            image = transforms.Normalize([0.5] * image_channel, [0.5] * image_channel)(image)
            pixels = image.permute(0, 2, 3, 1).reshape(-1, image_channel).to(device)
            coords = tuple(torch.linspace(-1, 1, steps=l) for l in image_size)
            coords = torch.stack(torch.meshgrid(*coords), dim=-1).reshape(-1, 2)
            coords = coords.repeat(batch_size, 1).view(batch_size, -1, 2).to(device)
            for params in outer_params:  # Inner loop
                params.requires_grad = False
            latent_params = torch.zeros((batch_size, *model.latent_dim)).to(device)
            latent_params.requires_grad = True
            meta_loss = []
            for inner_step in range(inner_num):
                modulations = model.l2m(latent_params)
                logits = model(coords, modulations)
                loss = criterion(logits, pixels) / image_dim
                latent_params = meta_sgd.update_weights(loss, latent_params)
                meta_loss.append(loss)
            meta_loss = torch.stack(meta_loss).mean()
            for params in outer_params:   # Outer update
                params.requires_grad = True
            latent_params = latent_params.detach()
            modulations = model.l2m(latent_params)
            logits = model(coords, modulations)
            loss = criterion(logits, pixels) / image_dim + meta_loss
            outer_optim.zero_grad()
            loss.backward()
            outer_optim.step()
            if not outer_step % summary_step:
                print("Epoch {} Outer step {}".format(e, outer_step))
                print("Train loss: {}".format(loss.item()))
                mpsnr = 0
                image_batch = pixels.cpu().view(batch_size, image_size[0], image_size[1], image_channel)
                predicted_batch = logits.cpu().view(batch_size, image_size[0], image_size[1], image_channel)
                for i in range(batch_size):
                    mpsnr += PSNR(256*(image_batch[i].detach().numpy() * 0.5 + 0.5), 256*(predicted_batch[i].detach().numpy() * 0.5 + 0.5))
                print("PSNR {}".format(mpsnr / batch_size))
                fig, axes = plt.subplots(1,2, figsize=(18, 6))
                axes[0].imshow(image_batch[i].detach().numpy() * 0.5 + 0.5, cmap=color)
                axes[1].imshow(predicted_batch[i].detach().numpy() * 0.5 + 0.5, cmap=color)
                plt.show()
        with open(optim_path, 'wb') as f:
            pickle.dump(meta_sgd, f)
        torch.save(model, save_path)