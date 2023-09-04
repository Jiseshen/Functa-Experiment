import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def functify(image, model, optim=None, loop=3):
    for param in model.parameters():
        param.requires_grad = False
    for siren_layer in model.siren_layers:
        for param in siren_layer.parameters():
            param.requires_grad = False
    criterion = torch.nn.MSELoss(reduction='mean')
    image = transforms.Normalize([0.5] * image.shape[0], [0.5] * image.shape[0])(image).to(device)
    pixels = image.permute(1, 2, 0).reshape(-1, image.shape[0])
    coords = tuple(torch.linspace(-1, 1, steps=l) for l in image.shape[1:])
    coords = torch.stack(torch.meshgrid(*coords), dim=-1).to(device).reshape(-1, 2).unsqueeze(0)
    latent = torch.zeros((1, *model.latent_dim)).to(device)
    latent.requires_grad = True
    if optim is None:
        optimizer = torch.optim.SGD([latent], lr=1e-2)
    for i in range(loop):
        modulations = model.l2m(latent)
        logits = model(coords, modulations)
        loss = criterion(logits, pixels)
        if optim is None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            latent = optim.update_weights(loss, latent)
    return latent[0].detach().cpu()


class Functaset(Dataset):
    def __init__(self, functa_list):
        super().__init__()
        self.functa_list = functa_list

    def __getitem__(self, index):
        return self.functa_list[index]

    def __len__(self):
        return len(self.functa_list)


def build_functaset(dataset, model, optim=None, loop=3):
    functa_list = []
    for image, label in dataset:
        functa_list.append((functify(image, model, optim, loop), label))
    print("Functaset Got!")
    print("Latent shape:")
    print(functa_list[0][0].shape)
    return Functaset(functa_list)


class Classifier(torch.nn.Module):
    def __init__(self, feature_dim, class_num, hidden_width=1024, hidden_depth=1):
        super().__init__()
        self.net = [torch.nn.Linear(feature_dim, hidden_width), torch.nn.SiLU(inplace=True)]
        for i in range(hidden_depth):
            self.net += [torch.nn.Linear(hidden_width, hidden_width), torch.nn.SiLU(inplace=True)]
        self.net += [torch.nn.Dropout(0.2), torch.nn.Linear(hidden_width, class_num)]
        self.net = torch.nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


class ConvClassifier(torch.nn.Module):
    def __init__(self, feature_size, feature_channel, class_num, hidden_channel=128, hidden_depth=1):
        super().__init__()
        self.net = [torch.nn.BatchNorm2d(feature_channel), torch.nn.Conv2d(feature_channel, hidden_channel, 3),
                    torch.nn.SiLU(inplace=True)]
        for i in range(hidden_depth):
            self.net += [torch.nn.BatchNorm2d(hidden_channel), torch.nn.Conv2d(hidden_channel, hidden_channel, 3),
                         torch.nn.SiLU(inplace=True)]
        self.net += [torch.nn.MaxPool2d(2), torch.nn.Flatten(), torch.nn.Dropout(0.2),
                     torch.nn.Linear(((feature_size - 4) / 2) ** 2 * hidden_channel, class_num)]
        self.net = torch.nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)


def train_classification(train_loader, test_loader, val_loader, model, epoch=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    for i in range(epoch):
        model.train()
        for functa, label in train_loader:
            logit = model(functa)
            loss = criterion(logit, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        acc = 0
        for functa, label in val_loader:
            logit = model(functa)
            acc += (logit.argmax(1) == label).type(torch.float).sum().item()
        print(acc / len(val_loader.dataset))
    model.eval()
    acc = 0
    for functa, label in test_loader:
        logit = model(functa)
        acc += (logit.argmax(1) == label).type(torch.float).sum().item()
    print(acc / len(test_loader.dataset))
