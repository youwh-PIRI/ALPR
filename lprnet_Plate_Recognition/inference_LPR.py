# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
test pretrained model.
Author: aiboy.wei@outlook.com .
'''

from lprnet_Plate_Recognition.data.load_data import CHARS, CHARS_DICT, LPRDataLoader
from PIL import Image, ImageDraw, ImageFont
from lprnet_Plate_Recognition.model.LPRNet import build_lprnet
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import cv2
import os
# D:\DDDDDDDDDDDDDDDDDDDOWNLOAD\PyQt_cam\pr_img\Q67876.jpg
def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--test_img_dirs', default="./pr_img/", help='the test images path')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--test_batch_size', default=1, help='testing batch size.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
    parser.add_argument('--show', default=True, type=bool, help='show test image and its predict result or not.')
    parser.add_argument('--pretrained_model', default='./lprnet_Plate_Recognition/weights/Final_LPRNet_model.pth', help='pretrained base model')

    args = parser.parse_args()

    return args

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

#
# def cv_imread(filePath):
#     cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
#     ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
#     ##cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)


args = get_parser()

lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
device = torch.device("cuda:0" if args.cuda else "cpu")
lprnet.to(device)
# print("Successful to build network!")

# load pretrained model
# if args.pretrained_model:
lprnet.load_state_dict(torch.load(args.pretrained_model,map_location='cpu'))
# print("loading pr model from {}!".format(args.pretrained_model))
print("load pr model !")



def inference():
    args1 = get_parser()

    # print('net:   ',lprnet)
    test_img_dirs = os.path.expanduser(args1.test_img_dirs)
    # print('dir  ',test_img_dirs)
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args1.img_size, args1.lpr_max_len)
    # print('len_testdataset:  ',len(test_dataset))
    # try:
    result = Greedy_Decode_Eval(lprnet, test_dataset, args1)
    # finally:
    #     cv2.destroyAllWindows()
    print("\tresult:",result)
    return result


def Greedy_Decode_Eval(Net, datasets, args):
    # TestNet = Net.eval()
    epoch_size = len(datasets) // args.test_batch_size
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    # Tp = 0
    # Tn_1 = 0
    # Tn_2 = 0
    # t1 = time.time()
    for i in range(epoch_size):
        # load train data
        # print("start")
        images, labels, lengths = next(batch_iterator)
        # start = 0
        # targets = []

        # for length in lengths:
        #     label = labels[start:start+length]
        #     targets.append(label)
        #     start += length
        # targets = np.array([el.numpy() for el in targets])
        # imgs = images.numpy().copy()

        if args.cuda:
            images = Variable(images.cuda())
        else:
            images = Variable(images)

        # forward
        prebs = Net(images)
        # greedy decode
        prebs = prebs.cpu().detach().numpy()
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j], axis=0))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label: # dropout repeate label and blank label
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        # print("pre:",preb_labels)
        for i, label in enumerate(preb_labels):
            # show  predict label
            result1 = show(label)
    # print("end")

    return result1
def show( label ):

    lb = ""
    for i in label:
        lb += CHARS[i]
    # tg = ""
    # for j in target.tolist():
    #     tg += CHARS[int(j)]

    # 可显示结果
    # img = cv2.putText(img, lb, (0,16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 255), 1)
    # img = cv2ImgAddText(img, lb, (0, 0))
    # cv2.imshow("test", img)
    return lb

# def cv2ImgAddText(img, text, pos, textColor=(255, 0, 0), textSize=12):
#     if (isinstance(img, np.ndarray)):  # detect opencv format or not
#         img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(img)
#     fontText = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
#     draw.text(pos, text, textColor, font=fontText)
#
#     return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


# if __name__ == "__main__":
#     inference()
