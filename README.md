# ALPR
基于4种轻量级深度卷积网络的无场景约束全自动车牌识别
车辆检测器控制系统开启状态，提前加载模型，补光灯控制光照，识别控制道闸
车辆检测+车辆分类+车牌定位+车牌修复(逆光,强光)+车牌识别
检测：yolov3_tiny(darknet)  有数据集计算iou即可，     强调速度
分类：b-cnn(resnet-18)                              强调准确率
定位：retina-net(mobilenetv1) 有数据集计算iou即可，   强调速度
修复：dcgan
ocr :lpr-net(cnn+ctc)    有数据集CCPD计算准确率即可， 强调速度和准确率
