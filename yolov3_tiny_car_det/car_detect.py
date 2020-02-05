from __future__ import division

from yolov3_tiny_car_det.models import *
from yolov3_tiny_car_det.utils.utils import *
from yolov3_tiny_car_det.utils.datasets import *

import os,cv2
import sys
import time
import datetime
import argparse
import  shutil
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
os.chdir(os.path.dirname(sys.argv[0]))
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default="", help="path to dataset")
parser.add_argument("--model_def", type=str, default="yolov3_tiny_car_det/config/yolov3-tiny.cfg",
                    help="path to model definition file")
parser.add_argument("--weights_path", type=str, default="yolov3_tiny_car_det/weights/yolov3-tiny.weights",
                    help="path to weights file")
parser.add_argument("--class_path", type=str, default="yolov3_tiny_car_det/data/coco.names",
                    help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.2, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
opt = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set up model
model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

if opt.weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(opt.weights_path)
else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path))

model.eval()  # Set in evaluation mode
# print("loading cd model from {}!".format(opt.weights_path))
print("load cd model !")

def yolo_car_det(myimage):
    opt = parser.parse_args()

    # print(opt)
    opt.image_folder = myimage

    os.makedirs("output", exist_ok=True)
    oneimg_patt = False
    # if ('jpg' in opt.image_folder):
    if ('jpg' in opt.image_folder) or ('JPG' in opt.image_folder) or ('PNG' in opt.image_folder) or ('png' in opt.image_folder):

        # print('one_img')
        oneimg_patt = True
        os.makedirs("new_test_img", exist_ok=True)
        # os.getcwd()
        new_file_fold = f'{os.getcwd()}\\new_test_img\\'
        new_file = f'{os.getcwd()}\\new_test_img\\0.jpg'
        # os.system(f"xcopy {opt.image_folder} {new_file_fold}")
        shutil.copyfile(opt.image_folder, new_file)
        opt.image_folder = new_file_fold

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    # print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))
        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # print("\nSaving images:")
    filename_all = list()
    # Iterate through images and save plot of detections

    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        # Create plot
        is_car_img = False
        print(img_i, path)
        # img = np.array(Image.open(path))
        roi = cv2.imread(path)
        # print('img_shape:',roi.shape)
        # plt.figure()
        # fig, ax = plt.subplots(1)
        # ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, roi.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            # print(type(detections))
            n_cls_preds = len(unique_labels)
            # print("2",type(detections))
            detections = detections.numpy()
            save_num = 0
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                save_num += 1
                is_car = False
                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                # print("坐标",x1, y1, x2, y2)
                if (classes[int(cls_pred)] in ['car','bus','truck']) & (float(cls_conf.item())>0.3):
                    is_car_img = True
                    is_car = True
                box_w = x2 - x1
                box_h = y2 - y1
                roiq = roi[int(y1):int(y2),int(x1):int(x2),:]
                # bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor='red', facecolor="none")
                if is_car:
                    if  not (((int(box_w)<400)&(int(box_h)<400)) or (float(box_h/box_w)>2)):
                    # if  is_car:
                        # plt.axis("off")
                        # plt.gca().xaxis.set_major_locator(NullLocator())
                        # plt.gca().yaxis.set_major_locator(NullLocator())
                        filename_m = f"car{time.strftime('%m-%d-%H-%M-%S',time.localtime(time.time()))}-{img_i}-{save_num}.jpg"
                        cv2.imwrite(f"yolo_output/{filename_m}",roiq)
                        # remove mix img
                        fsize = os.path.getsize(f'yolo_output/{filename_m}')
                        if (fsize < 40000):
                            os.remove(f'yolo_output/{filename_m}')
                            is_car_img = False
                        else:
                            filename_all.append(filename_m)
                    else:
                        is_car_img = False


        if is_car_img:
            print("has car")
        else:
            filename_all.append('none')

    print('filename',filename_all)
    if (oneimg_patt):
        return filename_all[0], is_car_img
    else:
        return filename_all, is_car_img

# print('res:',yolo_car_det('D:/DDDDDDDDDDDDDDDDDDDOWNLOAD/PyQt_cam/plate_location/test_images/2.jpg'))