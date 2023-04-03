from time import time
from turtle import back
from model import SLFIR_Model
import time
import torch
import numpy as np
import argparse
from dataset import *
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
GAMMA = 0.9
associate_weight = 1
# absolute path
tb_logdir = r"/home/ubuntu/lxd-workplace/LYT/face-sbir-new/1000/16/stage2/run/"
rgb_dir = r'/home/ubuntu/lxd-workplace/LYT/face-sbir-new/1000'
backbone_model_dir = r'/home/ubuntu/lxd-workplace/LYT/face-sbir-new/1000/16/stage1/InceptionV3_Face_backbone_best.pth'
attn_model_dir = r'/home/ubuntu/lxd-workplace/LYT/face-sbir-new/1000/16/stage1/InceptionV3_Face_attn_best.pth'
# rgb_dir = r'/home/ubuntu/lxd-workplace/LYT/face-sbir-new/1000'
# tb_logdir = r"./run/"
# backbone_model_dir = r'../stage1/InceptionV3_Face_backbone_best.pth'
# attn_model_dir = r'../stage1/InceptionV3_Face_attn_best.pth'

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
hp.lr = 0.0005
hp.epoches = 300
hp.save_iter = 50
hp.feature_num = 16

slfir_model = SLFIR_Model(hp)
dataloader_sketch_train, dataloader_sketch_test = get_dataloader(hp)

def main_train():
    mean_IOU_buffer = 0
    real_p = [0, 0, 0, 0]
    loss_buffer = []
    tb_writer = SummaryWriter(log_dir=tb_logdir)
    Top1_Song = [0]
    Top5_Song = [0]
    Top10_Song = [0]
    meanIOU_Song = []
    meanMA_Song = []
    meanOur_Song = []
    step_stddev = 0

    for epoch in range(hp.epoches):
        for i, sanpled_batch in enumerate(dataloader_sketch_train):
            start_time = time.time()
            loss_triplet = slfir_model.train_model(sanpled_batch)
            loss_buffer.append(loss_triplet)
            # 累加损失
            step_stddev += 1
            tb_writer.add_scalar('total loss', loss_triplet, step_stddev)
            print('epoch: {}, iter: {}, loss: {}, time cost{}'.format(epoch, step_stddev, loss_triplet, time.time()-start_time))

            # 模型预热20个epoch，然后开始隔几个batchsize测试
            if epoch >= 20 and step_stddev % hp.save_iter==0: #[evaluate after every 32*4 images]
                with torch.no_grad():
                    start_time = time.time()
                    top1, top5, top10, mean_IOU, mean_MA, mean_Our = slfir_model.evaluate_NN(dataloader_sketch_test)
                    slfir_model.train()
                    print('Epoch: {}, Iteration: {}:'.format(epoch, step_stddev))
                    print("TEST A@1: {}".format(top1))
                    print("TEST A@5: {}".format(top5))
                    print("TEST A@10: {}".format(top10))
                    print("TEST M@B: {}".format(mean_IOU))
                    print("TEST M@A: {}".format(mean_MA))
                    print("TEST Our: {}".format(mean_Our))
                    print("TEST Time: {}".format(time.time()-start_time))
                    Top1_Song.append(top1)
                    Top5_Song.append(top5)
                    Top10_Song.append(top10)
                    meanIOU_Song.append(mean_IOU)
                    meanMA_Song.append(mean_MA)
                    meanOur_Song.append(mean_Our)
                    tb_writer.add_scalar('TEST A@1', top1, step_stddev)
                    tb_writer.add_scalar('TEST A@5', top5, step_stddev)
                    tb_writer.add_scalar('TEST A@10', top10, step_stddev)
                    tb_writer.add_scalar('TEST M@B', mean_IOU, step_stddev)
                    tb_writer.add_scalar('TEST M@A', mean_MA, step_stddev)
                    tb_writer.add_scalar('TEST Our', mean_Our, step_stddev)

                if mean_IOU > mean_IOU_buffer:
                    torch.save(slfir_model.backbone_network.state_dict(), 'model' + '_backbone_' + '.pth')
                    torch.save(slfir_model.attn_network.state_dict(), 'model' + '_attn_' + '.pth')
                    torch.save(slfir_model.lstm_network.state_dict(), 'model' + '_lstm_' + '.pth')
                    mean_IOU_buffer = mean_IOU

                    # # 这种做法会导致其他指标偏高
                    real_p = [top1, top5, top10, mean_MA]
                    # 更改后符合保存模型时的真实指标
                    print('Model Updated')

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
    print("meanOur_MAX: {}".format(max(meanOur_Song)))
    print(Top1_Song)
    print(Top5_Song)
    print(Top10_Song)
    print(meanIOU_Song)
    print(meanMA_Song)
    print(meanOur_Song)


if __name__ == "__main__":
    main_train()




