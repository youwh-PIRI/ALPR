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
![图片名称可省略.jpg](https://github.com/youwh-PIRI/ALPR/blob/master/img/ui.JPG)

更换车辆检测网络为仅使用yolov3_tiny(pytorch版本)

分类功能暂时不做，修复功能待完善

#检测对应文件夹：yolov3_tiny_car_det

|  size  |gtx titan x(fps)|
| ------ | -------------- |
|  33M   |       220      |

#定位对应文件夹：plate_location

利用[RetinaFace](https://github.com/biubug6/Pytorch_Retinaface)进行迁移学习实现的车牌检测、车牌四角定位、车牌矫正对齐

可参考我的博客链接，RETINAFACE论文笔记

>https://www.cnblogs.com/ywheunji/p/12285421.html

## 数据

**地址**

 [https://github.com/detectRecog/CCPD](https://github.com/detectRecog/CCPD)

## performance

使用mobilenet0.25作为骨干网时，模型大小仅为1.7M

|  size  |inference@gtx 1060(ms)|
| ------ | -------------------- |
|  1.7M  |       0.5-           |

#识别对应文件夹：lprnet_Plate_Recognition

可参考我的博客链接，lprnet论文笔记

>https://www.cnblogs.com/ywheunji/p/12268340.html

## performance

- include blue/green license plate.
- test images number is 27320.

|  size  | personal test imgs(%) | inference@gtx 1060(ms) |
| ------ | --------------------- | ---------------------- |
|  1.7M  |         96.0+         |          0.5-          |

#环境：
- python3
- openv-python 3.x
- pytorch 1.1
- pyqt5.9
- imutils
- Pillow

#打开可视化界面：

直接运行camershow.py文件即可进入pyqt的ui界面，支持视频，单张，摄像头模式


目前完成状态：

基本功能建造完成，待优化cpu版本的速度

优化暂无时间做，若追求速度，可直接把车辆检测去掉，基本不影响准确率，可达到低性能设备实时检测效果

## 参考文献


>http://openaccess.thecvf.com/content_ECCV_2018/papers/Sergio_Silva_License_Plate_Detection_ECCV_2018_paper.pdf

>https://github.com/lyl8213/Plate_Recognition-LPRnet

>https://github.com/gm19900510/Pytorch_Retina_License_Plate

>https://github.com/CaptainEven/Vehicle-Car-detection-and-multilabel-classification

>https://blog.csdn.net/weixin_43008870/article/details/86496263

>http://www.szjiuding.com/237.html

>https://github.com/eriklindernoren/PyTorch-YOLOv3





