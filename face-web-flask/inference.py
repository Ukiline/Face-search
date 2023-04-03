from time import time
from model import SLFIR_Model
from dataset import *
import torch
import numpy as np
import argparse
from PIL import Image
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
associate_weight = 1
rgb_dir = r'photo'
backbone_model_dir = r'model_backbone_.pth'
attn_model_dir = r'model_attn_.pth'
lstm_model_dir = r'model_lstm_.pth'

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
hp = parser.parse_args()
hp.root_dir = rgb_dir
hp.nThreads = 4
hp.device = device
hp.backbone_model_dir = backbone_model_dir
hp.attn_model_dir = attn_model_dir
# hp.backbone_lr = 0.0005
hp.batchsize = 32
hp.lr = 0.00005
hp.epoches = 300
hp.save_iter = 20
hp.feature_num = 16

# self.model = SLFIR_Model(hp)
# dataloader_sketch_test = get_dataloader(hp)

class Inference():
    def __init__(self):
        self.dataloader_sketch_test = get_dataloader(hp)
        self.model = SLFIR_Model(hp)
        self.model.backbone_network.load_state_dict(torch.load(backbone_model_dir, map_location=device))
        self.model.backbone_network.to(device)
        self.model.backbone_network.fixed_param()
        self.model.backbone_network.eval()

        self.model.attn_network.load_state_dict(torch.load(attn_model_dir, map_location=device))
        self.model.attn_network.to(device)
        self.model.attn_network.fixed_param()
        self.model.attn_network.eval()

        self.model.lstm_network.load_state_dict(torch.load(lstm_model_dir, map_location=device))
        self.model.lstm_network.to(device)
        for param in self.model.lstm_network.parameters():
            param.requires_grad = False
        self.model.lstm_network.eval()

        # 提前获取正样本的向量
        self.Image_List = []
        self.Image_Path_List = []
        for idx, batch in enumerate(self.dataloader_sketch_test):
            positive_feature = self.model.lstm_network(self.model.attn_network(
                self.model.backbone_network(batch['positive_img'].to(device))))
            self.Image_List.append(positive_feature)
            self.Image_Path_List.append(batch['positive_path'])
        self.Image_List = torch.stack(self.Image_List).view(len(self.Image_List), -1)

    # 获取草图的向量
    def get_sketch_vector(self, img_tensor):
        return self.model.lstm_network(self.model.attn_network(
                self.model.backbone_network(img_tensor.to(device))))

    # 草图预处理并转换为tensor
    def sketch_preprocess(self, sketch_img):
        def get_transform():
            transform_list = []
            transform_list.extend([transforms.Resize(299)])
            transform_list.extend(
                [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            return transforms.Compose(transform_list)
        sketch_img = Image.fromarray(sketch_img).convert('RGB')
        return get_transform()(sketch_img)
        
    def get_rankList(self, sketch_img):
        def SortNameByData(dataList, nameList):
            convertDic = {}
            sortedDic = {}
            sortedNameList = []
            for index in range(len(dataList)):
                convertDic[index] = dataList[index]
            sortedDic = sorted(convertDic.items(), key=lambda item: item[1], reverse=False)
            for key, _ in sortedDic:
                sortedNameList.append(nameList[key])
            return sortedNameList
        sketch_tensor = self.sketch_preprocess(sketch_img)
        sketch_feature = self.get_sketch_vector(sketch_tensor.unsqueeze(0))
        distance = F.pairwise_distance(F.normalize(sketch_feature.to(device)), self.Image_List.to(device))
        rankingList = SortNameByData(distance, self.Image_Path_List)
        return rankingList

    def inference(self, img):
        with torch.no_grad():
            sortedName = self.get_rankList(img)
        return sortedName




