from time import time
from eval_model import SLFIR_Model
import time
import torch
import numpy as np
import argparse
from dataset import *
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
GAMMA = 0.9
associate_weight = 1
rgb_dir = r'/home/ubuntu/lxd-workplace/LYT/face-sbir-new/1000'
backbone_model_dir = r'/home/ubuntu/lxd-workplace/LYT/face-sbir-new/1000/16/stage2/model_backbone_.pth'
attn_model_dir = r'/home/ubuntu/lxd-workplace/LYT/face-sbir-new/1000/16/stage2/model_attn_.pth'
lstm_model_dir = r'/home/ubuntu/lxd-workplace/LYT/face-sbir-new/1000/16/stage2/model_lstm_.pth'

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

slfir_model = SLFIR_Model(hp)
dataloader_sketch_train, dataloader_sketch_test = get_dataloader(hp)

def fixed_network():
    slfir_model.backbone_network.load_state_dict(torch.load(backbone_model_dir, map_location=device))
    slfir_model.backbone_network.to(device)
    slfir_model.backbone_network.fixed_param()
    slfir_model.backbone_network.eval()

    slfir_model.attn_network.load_state_dict(torch.load(attn_model_dir, map_location=device))
    slfir_model.attn_network.to(device)
    slfir_model.attn_network.fixed_param()
    slfir_model.attn_network.eval()

    slfir_model.lstm_network.load_state_dict(torch.load(lstm_model_dir, map_location=device))
    slfir_model.lstm_network.to(device)
    for param in slfir_model.lstm_network.parameters():
        param.requires_grad = False
    slfir_model.lstm_network.eval()

def main_eval():
    mean_IOU_buffer = 0
    real_p = [0, 0, 0, 0]
    Top1_Song = [0]
    Top5_Song = [0]
    Top10_Song = [0]
    meanIOU_Song = []
    meanMA_Song = []

    fixed_network()

    with torch.no_grad():
        start_time = time.time()
        top1, top5, top10, mean_IOU, mean_MA, mean_OurB, mean_OurA = slfir_model.evaluate_NN(dataloader_sketch_test)
        print("TEST A@1: {}".format(top1))
        print("TEST A@5: {}".format(top5))
        print("TEST A@10: {}".format(top10))
        print("TEST M@B: {}".format(mean_IOU))
        print("TEST M@A: {}".format(mean_MA))
        print("TEST OurB: {}".format(mean_OurB))
        print("TEST OurA: {}".format(mean_OurA))
        print("TEST Time: {}".format(time.time()-start_time))
        Top1_Song.append(top1)
        Top5_Song.append(top5)
        Top10_Song.append(top10)
        meanIOU_Song.append(mean_IOU)
        meanMA_Song.append(mean_MA)

    # print('Best at MB: Top1: {}, Top5: {}, Top10: {}, MB: {}, MA: {},'.format(top1_buffer, top5_buffer,
    #                                                                           top10_buffer, mean_IOU_buffer,
    #                                                                           mean_MA_buffer))
    print('REAL performance: Top1: {}, Top5: {}, Top10: {}, MB: {}, MA: {},'.format(real_p[0], real_p[1],
                                                                                    real_p[2],
                                                                                    mean_IOU_buffer,
                                                                                    real_p[3]))


    print("TOP1_MAX: {}".format(max(Top1_Song)))
    print("TOP5_MAX: {}".format(max(Top5_Song)))
    print("TOP10_MAX: {}".format(max(Top10_Song)))
    print("meaIOU_MAX: {}".format(max((meanIOU_Song))))
    print("meaMA_MAX: {}".format(max((meanMA_Song))))
    print(Top1_Song)
    print(Top5_Song)
    print(Top10_Song)
    print(meanIOU_Song)
    print(meanMA_Song)


if __name__ == "__main__":
    main_eval()




