import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt, dsbopt, dsbpredopt
from data.dataset import Dataset, TestDataset, inverse_normalize, DSBDataset, DSBTestDataset, DSBPredictDataset
from model import FasterRCNNVGG16
from torch.autograd import Variable
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

import pandas as pd
import numpy as np

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x):
    lab_img = x
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0], sizes[1][0]]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    dsbopt._parse(kwargs)

    dataset = DSBDataset(dsbopt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=dsbopt.num_workers)
    testset = DSBTestDataset(dsbopt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=dsbopt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if dsbopt.load_path:
        trainer.load(dsbopt.load_path)
        print('load pretrained model from %s' % dsbopt.load_path)

    trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = dsbopt.lr
    for epoch in range(dsbopt.epoch):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            img, bbox, label = Variable(img), Variable(bbox), Variable(label)
            trainer.train_step(img, bbox, label, scale)

            if (ii + 1) % dsbopt.plot_every == 0:
                if os.path.exists(dsbopt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        eval_result = eval(test_dataloader, faster_rcnn, test_num=dsbopt.test_num)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 50:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(dsbopt.lr_decay)
            lr_ = lr_ * dsbopt.lr_decay

        trainer.vis.plot('test_map', eval_result['map'])
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)
        #if epoch == 13: 
        #    break

def predict(**kwargs):
    dsbpredopt._parse(kwargs)

    dataset = DSBPredictDataset(dsbpredopt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=False, \
                                  pin_memory=True,
                                  num_workers=dsbpredopt.num_workers)
                                  
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if dsbpredopt.load_path:
        trainer.load(dsbpredopt.load_path)
        print('load checkpoint from %s' % dsbpredopt.load_path)
        
    new_test_ids = []
    rles = []

    for ii, (imgs, sizes, predicted_mask, id_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0], sizes[1][0]]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        #pred_img = visdom_bbox(at.tonumpy(imgs[0]),
        #                       at.tonumpy(pred_bboxes_[0]),
        #                       at.tonumpy(pred_labels_[0]).reshape(-1),
        #                       at.tonumpy(pred_scores_[0]))
        #pred_mask_img = visdom_bbox(at.tonumpy(predicted_mask[0]),
        #                       at.tonumpy(pred_bboxes_[0]),
        #                       at.tonumpy(pred_labels_[0]).reshape(-1),
        #                       at.tonumpy(pred_scores_[0]))
                               
        #trainer.vis.img('pred_img', pred_img)
        #trainer.vis.img('pred_mask_img', pred_mask_img)
        #input("Press Enter to continue...")
        
        predicted_mask_labeled = np.squeeze(at.tonumpy(predicted_mask[0]).copy())
        pred_bboxes_ = at.tonumpy(pred_bboxes_[0]).astype(np.uint16)

        if pred_bboxes_.shape[0] == 0:
            print(id_[0])

        if predicted_mask_labeled.shape[0] != sizes[0] or predicted_mask_labeled.shape[1] != sizes[1]:
            print('wtf')

        for idx, pred_bbox in enumerate(pred_bboxes_):
            mask = predicted_mask_labeled[pred_bbox[0]:pred_bbox[2], pred_bbox[1]:pred_bbox[3]]
            #print(predicted_mask_labeled.shape)
            #print(pred_bbox[0])
            #print(pred_bbox[2])
            #print(pred_bbox[1])
            #print(pred_bbox[3])
            #print(mask)
            #input("input")
            if (pred_bbox[2] > sizes[0] or pred_bbox[3] > sizes[1]):
                print('wtf')
            mask[mask > 0] = idx+1
            predicted_mask_labeled[pred_bbox[0]:pred_bbox[2], pred_bbox[1]:pred_bbox[3]] = mask
        
        predicted_mask_labeled[predicted_mask_labeled == 255] = 0
        #print(predicted_mask_labeled)
        rle = list(prob_to_rles(predicted_mask_labeled))
        #print(rle)
        #exit()
        #for r_i, rle_i in enumerate(rle):
            #for r_j, rle_j in enumerate(rle_i):
                #if r_j % 2 == 0:
                    #if (rle_j-1)%sizes[0]+1+rle_i[r_j+1]-1 >= sizes[0]:
                        #print(rle_j)
                        #print(rle_i[r_j+1])
                        #print((rle_j-1)%sizes[0]+1+rle_i[r_j+1]-1)
                        #print(sizes[0])
                        #print('out of size 0')
                        #print(rle[r_i][r_j+1])
                        #print(r_i)
                        #print(r_j+1)
                        #rle[r_i][r_j+1] = rle[r_i][r_j+1] - 1
                        #print(rle[r_i][r_j+1])
                    #if rle_j + rle_i[r_j+1]-1 >= sizes[0] * sizes[1]:
                        #print('out of total number')
        #rle[0][1] = 6
        #print(rle)
        #exit()
        rles.extend(rle)
        new_test_ids.extend([id_[0]] * len(rle))
        

    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv("predicts/unet__data_0_d6_t_c_lr9_bs4_size256_epoch_74.csv", index=False)

if __name__ == '__main__':
    import fire

    fire.Fire()
