from time import time
from model import SLFIR_Model
import time
import torch
import numpy as np
import argparse
from dataset_eval_100 import *
import torch.nn.functional as F

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
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
dataloader_sketch_test = get_dataloader(hp)

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

def evaluate_NN(slfir_model, dataloader):
        slfir_model.backbone_network.eval()
        slfir_model.attn_network.eval()
        slfir_model.lstm_network.eval()

        slfir_model.Sketch_Array_Test = []
        slfir_model.Image_Array_Test = []
        slfir_model.Pixel_Ratio = []
        for idx, batch in enumerate(dataloader):
            sketch_feature = slfir_model.attn_network(
                slfir_model.backbone_network(batch['sketch_seq'].squeeze(0).to(device)))
            positive_feature = slfir_model.lstm_network(slfir_model.attn_network(
                slfir_model.backbone_network(batch['positive_img'].to(device))))
            slfir_model.Sketch_Array_Test.append(sketch_feature)
            slfir_model.Image_Array_Test.append(positive_feature)
            slfir_model.Pixel_Ratio.append(batch['pixel_ratio'])
        # slfir_model.Sketch_Array_Test = torch.stack(slfir_model.Sketch_Array_Test)
        slfir_model.Image_Array_Test = torch.stack(slfir_model.Image_Array_Test)
        slfir_model.Image_Array_Test = slfir_model.Image_Array_Test.view(slfir_model.Image_Array_Test.shape[0], -1)
        # num_of_Sketch_Step = len(slfir_model.Sketch_Array_Test[0])
        avererage_area = []
        avererage_area_percentile = []
        avererage_ourB = []
        avererage_ourA = []
        # exps = np.linspace(1,num_of_Sketch_Step, num_of_Sketch_Step) / num_of_Sketch_Step
        # factor = np.exp(1 - exps) / np.e
        rank_all = []
        top1_accuracy = 0
        top5_accuracy = 0
        top10_accuracy = 0
        for i_batch, sanpled_batch in enumerate(slfir_model.Sketch_Array_Test):
            mean_rank = []
            mean_rank_percentile = []
            mean_rank_ourB = []
            mean_rank_ourA = []
            num_of_Sketch_Step = len(sanpled_batch)
            rank = torch.zeros(num_of_Sketch_Step)
            rank_percentile = torch.zeros(num_of_Sketch_Step)
            exps = slfir_model.Pixel_Ratio[i_batch].view(-1).numpy()
            factor = np.exp(1 - exps) / np.e
            for i_sketch in range(sanpled_batch.shape[0]):
                sketch_feature = slfir_model.lstm_network(sanpled_batch[:i_sketch+1].to(device))
                target_distance = F.pairwise_distance(F.normalize(sketch_feature[-1].unsqueeze(0).to(device)), slfir_model.Image_Array_Test[i_batch].unsqueeze(0).to(device))
                distance = F.pairwise_distance(F.normalize(sketch_feature[-1].unsqueeze(0).to(device)), slfir_model.Image_Array_Test.to(device))
                #rankingList = self.SortNameByData(distance, self.Image_Name_Test)
                rank[i_sketch] = distance.le(target_distance).sum()
                #a.le(b),，若a<=b，返回1
                #.sum， 算出来直接等于rank
                rank_percentile[i_sketch] = (len(distance) - rank[i_sketch]) / (len(distance) - 1)
                #(len-rank)/(len-1)
                if rank[i_sketch].item() == 0:
                    #并不存在sum=0的情况，无用？
                    mean_rank.append(1.)
                else:
                    mean_rank.append(1/rank[i_sketch].item())
                    #1/(rank)
                    mean_rank_percentile.append(rank_percentile[i_sketch].item())
                    mean_rank_ourB.append(1/rank[i_sketch].item() * factor[i_sketch])
                    mean_rank_ourA.append(rank_percentile[i_sketch].item() * factor[i_sketch])
                    #rank_percentile
            avererage_area.append(np.sum(mean_rank)/len(mean_rank))
            avererage_area_percentile.append(np.sum(mean_rank_percentile)/len(mean_rank_percentile))
            avererage_ourB.append(np.sum(mean_rank_ourB)/len(mean_rank_ourB))
            avererage_ourA.append(np.sum(mean_rank_ourA)/len(mean_rank_ourA))
            rank_all.append(rank)
            top1_accuracy += rank[-1].le(1).sum().numpy()
            top5_accuracy += rank[-1].le(5).sum().numpy()
            top10_accuracy += rank[-1].le(10).sum().numpy()

        top1_accuracy = top1_accuracy / len(rank_all)
        top5_accuracy = top5_accuracy / len(rank_all)
        top10_accuracy = top10_accuracy / len(rank_all)
        #A@1 A@5 A%10
        meanIOU = np.mean(avererage_area)
        meanMA = np.mean(avererage_area_percentile)
        meanOurB = np.mean(avererage_ourB)
        meanOurA = np.mean(avererage_ourA)

        return top1_accuracy, top5_accuracy, top10_accuracy, meanIOU, meanMA, meanOurB, meanOurA

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
        top1, top5, top10, mean_IOU, mean_MA, mean_OurB, mean_OurA = evaluate_NN(slfir_model, dataloader_sketch_test)
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




