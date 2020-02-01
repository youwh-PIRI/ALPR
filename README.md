# ALPR

# 基于4种轻量级深度卷积网络的无场景约束全自动车牌识别

车辆检测器控制系统开启状态，提前加载模型，补光灯控制光照，识别控制道闸

## 车辆检测+车辆分类+车牌定位+车牌修复（强光）+车牌识别

### 检测：yolov3_tiny（darknet）突出速度

### 分类：b-cnn（resnet-18）突出准确率

### 定位：retina-net（mobilenetv1）突出速度

### 修复：dcgan(TODO)

### 识别：lpr-net（cnn + ctc）速度与准确率

本次提交以pyqt5界面形式展示

更换车辆检测网络为仅使用yolov3_tiny(pytorch版本)

分类功能暂时不做，修复功能待完善

检测对应文件夹：yolov3_tiny_car_det
定位对应文件夹：plate_location
识别对应文件夹：lprnet_Plate_Recognition
多余文件夹为测试所用



目前完成状态：
单张图片正常识别没问题
视频模式正在测试
摄像头模式未开始


## 参考文献


[ECCV-alpr-paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Sergio_Silva_License_Plate_Detection_ECCV_2018_paper.pdf)

>https://github.com/lyl8213/Plate_Recognition-LPRnet

>https://github.com/gm19900510/Pytorch_Retina_License_Plate

>https://github.com/CaptainEven/Vehicle-Car-detection-and-multilabel-classification

>https://blog.csdn.net/weixin_43008870/article/details/86496263

>http://www.szjiuding.com/237.html

>https://github.com/eriklindernoren/PyTorch-YOLOv3
