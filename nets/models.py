import torch.nn as nn
import clip
from utils.clip_util import freeze_param, get_image_features
import torch
import torch.nn.functional as F

class ClipModelat(nn.Module):

    CLIP_MODELS = [
        'RN50',
        'RN101',
        'RN50x4',
        'RN50x16',
        'RN50x64',
        'ViT-B/32',
        'ViT-B/16',
        'ViT-L/14',
        'ViT-L/14@336px'
    ]

    def __init__(self, model_name='Vit-B/32', device='cuda', logger=None, attention=True, freezepy=True):
        super(ClipModelat, self).__init__()
        self.logger = logger
        if type(model_name) is int:
            model_name = self.index_to_model(model_name)
        self.model, self.preprocess = clip.load(
            model_name, device=device, jit=False)
        self.model.eval()
        self.model.to(device)
        self.model_name = model_name
        self.attention = attention
        self.freezepy = freezepy
        self.device = device

    def initdgatal(self, dataloader):

        for batch in dataloader:
            image, _, label = batch
            image = image.to(self.device)
            label = label.to(self.device)
            image_features = get_image_features(
                image, self.model, self.preprocess)
            break
        if self.freezepy:
            freeze_param(self.model)

        def Global_attention_block(C_A):
            conv = nn.Conv2d(1, 1, kernel_size=1, padding='same').to('cuda')
            x = torch.mean(C_A, dim=1, keepdim=True)
            y, _ = torch.max(C_A, dim=1, keepdim=True)
            print(x.shape, y.shape)
            z = torch.cat((x, y), dim=1)
            z = F.relu(z)
            z = conv(z)
            attention_map = torch.sigmoid(z)
            # S_A = attention_map * C_A

            return attention_map

        if self.attention:
            # For FAA-CLIP
            self.fea_attn = nn.Sequential(nn.Linear(image_features.shape[1], image_features.shape[1]), nn.BatchNorm1d(image_features.shape[1]), nn.LeakyReLU(
            ), nn.Linear(image_features.shape[1], image_features.shape[1]), nn.Softmax(dim=1)).to(self.device)
            # self.fea_attn = nn.Sequential(nn.Linear(image_features.shape[1], image_features.shape[1]),
            #                                nn.Softmax(dim=1)).to(self.device)
            # self.fea_attn = nn.Sequential(nn.Linear(image_features.shape[1], image_features.shape[1]),
            #                               nn.BatchNorm1d(image_features.shape[1]),nn.Softmax(dim=1)).to(self.device)
            # self.fea_attn = nn.Sequential(nn.Linear(image_features.shape[1], image_features.shape[1]),
            #                               nn.BatchNorm1d(image_features.shape[1]), nn.LeakyReLU(
            #     ), nn.Softmax(dim=1)).to(self.device)
            # self.fea_attn = nn.Sequential(nn.Linear(image_features.shape[1], image_features.shape[1]),
            #                                nn.LeakyReLU(
            #     ), nn.Linear(image_features.shape[1], image_features.shape[1]), nn.Softmax(dim=1)).to(self.device)
            # self.fea_attn = Global_attention_block(image_features)
            # For FedClip
            # self.fea_attn = nn.Sequential(nn.Linear(image_features.shape[1], image_features.shape[1]),
            # nn.Tanh(), nn.Linear(image_features.shape[1], image_features.shape[1]), nn.Softmax(dim=1)).to(self.device)





    def index_to_model(self, index):
        return self.CLIP_MODELS[index]

    @staticmethod
    def get_model_name_by_index(index):
        name = ClipModelat.CLIP_MODELS[index]
        name = name.replace('/', '_')
        return name

    def setselflabel(self, labels):
        self.labels = labels