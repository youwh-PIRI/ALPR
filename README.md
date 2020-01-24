# ALPR

# 基于4种轻量级深度卷积网络的无场景约束全自动车牌识别

车辆检测器控制系统开启状态，提前加载模型，补光灯控制光照，识别控制道闸

## 车辆检测+车辆分类+车牌定位+车牌修复（强光）+车牌识别

### 检测：yolov3_tiny（darknet）突出速度

### 分类：b-cnn（resnet-18）突出准确率

### 定位：retina-net（mobilenetv1）突出速度

### 修复：dcgan(TODO)

### 识别：lpr-net（cnn + ctc）速度与准确率

## 参考文献

>https://github.com/lyl8213/Plate_Recognition-LPRnet

>https://github.com/gm19900510/Pytorch_Retina_License_Plate

>https://github.com/CaptainEven/Vehicle-Car-detection-and-multilabel-classification
