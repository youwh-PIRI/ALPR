from my_alpr import Ui_MainWindow
import sys
from PyQt5.QtWidgets import QApplication,QMainWindow,QFileDialog
from PyQt5.QtCore import QTimer,QCoreApplication
from PyQt5.QtGui import QPixmap
import cv2
import qimage2ndarray
import time,os,shutil
import numpy as np
import datetime,threading
import plate_location.pll_detect as pll
import yolov3_tiny_car_det.car_detect as ycd
import lprnet_Plate_Recognition.inference_LPR as plr

class CamShow(QMainWindow,Ui_MainWindow):
    def __del__(self):
        try:
            self.camera.release()  # 释放资源
        except:
            return
    def __init__(self,parent=None):
        super(CamShow,self).__init__(parent)
        self.setupUi(self)
        self.PrepSliders()
        self.PrepWidgets()
        self.PrepParameters()
        self.CallBackFunctions()
        self.Timer=QTimer()
        self.Timer.timeout.connect(self.TimerOutFun)

    def PrepSliders(self):
        self.RedColorSld.valueChanged.connect(self.RedColorSpB.setValue)
        self.RedColorSpB.valueChanged.connect(self.RedColorSld.setValue)
        self.GreenColorSld.valueChanged.connect(self.GreenColorSpB.setValue)
        self.GreenColorSpB.valueChanged.connect(self.GreenColorSld.setValue)
        self.BlueColorSld.valueChanged.connect(self.BlueColorSpB.setValue)
        self.BlueColorSpB.valueChanged.connect(self.BlueColorSld.setValue)
        self.ExpTimeSld.valueChanged.connect(self.ExpTimeSpB.setValue)
        self.ExpTimeSpB.valueChanged.connect(self.ExpTimeSld.setValue)
        self.GainSld.valueChanged.connect(self.GainSpB.setValue)
        self.GainSpB.valueChanged.connect(self.GainSld.setValue)
        self.BrightSld.valueChanged.connect(self.BrightSpB.setValue)
        self.BrightSpB.valueChanged.connect(self.BrightSld.setValue)
        self.ContrastSld.valueChanged.connect(self.ContrastSpB.setValue)
        self.ContrastSpB.valueChanged.connect(self.ContrastSld.setValue)
    def PrepWidgets(self):
        self.PrepCamera()
        self.StopBt.setEnabled(False)
        self.RecordBt.setEnabled(False)
        self.GrayImgCkB.setEnabled(False)
        self.RedColorSld.setEnabled(False)
        self.RedColorSpB.setEnabled(False)
        self.GreenColorSld.setEnabled(False)
        self.GreenColorSpB.setEnabled(False)
        self.BlueColorSld.setEnabled(False)
        self.BlueColorSpB.setEnabled(False)
        self.ExpTimeSld.setEnabled(False)
        self.ExpTimeSpB.setEnabled(False)
        self.GainSld.setEnabled(False)
        self.GainSpB.setEnabled(False)
        self.BrightSld.setEnabled(False)
        self.BrightSpB.setEnabled(False)
        self.ContrastSld.setEnabled(False)
        self.ContrastSpB.setEnabled(False)
    def PrepCamera(self):
        try:
            self.camera=cv2.VideoCapture(0)
            self.MsgTE.clear()
            self.MsgTE.append('Oboard camera connected.')
            self.MsgTE.setPlainText()
        except Exception as e:
            self.MsgTE.clear()
            self.MsgTE.append(str(e))
    def PrepParameters(self):
        self.RecordFlag=0
        self.RecordPath='img_save_path/'
        self.FilePathLE.setText(self.RecordPath)
        self.Image_num=0
        self.R=1
        self.G=1
        self.B=1

        self.ExpTimeSld.setValue(self.camera.get(15))
        self.SetExposure()
        self.GainSld.setValue(self.camera.get(14))
        self.SetGain()
        self.BrightSld.setValue(self.camera.get(10))
        self.SetBrightness()
        self.ContrastSld.setValue(self.camera.get(11))
        self.SetContrast()
        self.MsgTE.clear()
    def CallBackFunctions(self):
        self.FilePathBt.clicked.connect(self.SetFilePath)
        self.ShowBt.clicked.connect(self.StartCamera)
        self.StopBt.clicked.connect(self.StopCamera)
        self.RecordBt.clicked.connect(self.RecordCamera)
        # self.ExitBt.clicked.connect(self.ExitApp)
        self.GrayImgCkB.stateChanged.connect(self.SetGray)
        self.ExpTimeSld.valueChanged.connect(self.SetExposure)
        self.GainSld.valueChanged.connect(self.SetGain)
        self.BrightSld.valueChanged.connect(self.SetBrightness)
        self.ContrastSld.valueChanged.connect(self.SetContrast)
        self.RedColorSld.valueChanged.connect(self.SetR)
        self.GreenColorSld.valueChanged.connect(self.SetG)
        self.BlueColorSld.valueChanged.connect(self.SetB)
        self.img_pat.clicked.connect(self.tanchu1)
        self.vid_pat.clicked.connect(self.tanchu2)
        self.vid_pat.clicked.connect(self.choose_video)
        self.cam_pat.clicked.connect(self.tanchu3)
        self.img_pat.clicked.connect(self.choose_img)
        self.remove_file.clicked.connect(self.remove_img)
        self.mytest.clicked.connect(self.myte)

    def myte(self):
        num = 5
        while(num):
        # for i in range(5):
            num -= 1
            time.sleep(1)
            self.MsgTE.setText(f"{num}")


    def tanchu1(self):
        self.pattle.setText('单帧模式')
    def tanchu2(self):
        self.pattle.setText('视频模式')
    def tanchu3(self):
        self.pattle.setText('摄像头模式')
    def remove_img(self):
        for files in os.listdir('output'):
            if files.endswith(".jpg"):
                os.remove(os.path.join('output', files))

        for files in os.listdir('yolov3_tiny_car_det/output'):
            if files.endswith(".jpg"):
                os.remove(os.path.join('yolov3_tiny_car_det/output', files))
        self.MsgTE.setText("清除缓存！")

    # def remove_img(self):
    #
    #     del_list = os.listdir('output')
    #     for f in del_list:
    #         file_path = os.path.join('output', f)
    #         if os.path.isfile(file_path):
    #             os.remove(file_path)
    #         elif os.path.isdir(file_path):
    #             shutil.rmtree(file_path)
    #     del_list = os.listdir('output')
    #     for f in del_list:
    #         file_path = os.path.join('output', f)
    #         if os.path.isfile(file_path):
    #             os.remove(file_path)
    #         elif os.path.isdir(file_path):
    #             shutil.rmtree(file_path)

    def display1(self,num):
        ret, frame = self.videoCapture.read()
        # print(num)
        self.MsgTE.clear()
        self.MsgTE.setText(f"{num}")
        self.MsgTE.show()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = qimage2ndarray.array2qimage(frame)
        self.img_one.setPixmap(QPixmap(qimg).scaled(self.img_one.width(), self.img_one.height()))
        self.img_one.show()

    def choose_video(self):
        vidName, vidType = QFileDialog.getOpenFileName(self, "打开视频", "", "*.mp4;;*.avi;;All Files(*)")
        self.videoCapture = cv2.VideoCapture()
        self.videoCapture.open(vidName)
        print("视频位置：：",vidName)
        self.MsgTE.setText(f"视频位置：{vidName}")
        # 创建视频显示线程
        th = threading.Thread(target=self.choose_video1())
        th.start()

        # self.choose_video1()
    def choose_video1(self):
        # vidName, vidType = QFileDialog.getOpenFileName(self, "打开视频", "", "*.mp4;;*.avi;;All Files(*)")
        # # 创建视频显示线程
        # th = threading.Thread(target=self.Display)
        # th.start()

        self.videoCapture = cv2.VideoCapture()
        self.videoCapture.open(r'vid_te.mp4')

        fps = self.videoCapture.get(cv2.CAP_PROP_FPS)
        frames = self.videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        print("fps=", fps, "frames=", frames)
        while self.videoCapture.isOpened():
            ret, frame = self.videoCapture.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                qimg = qimage2ndarray.array2qimage(frame)
                self.img_one.setPixmap(QPixmap(qimg).scaled(self.img_one.width(), self.img_one.height()))
                self.img_one.show()

                cv2.waitKey(int(1000 / fps))
            else:
                break
        print('show_end')
        # for i in range(int(frames)):
        #
        #     self.display1(i)

            # if i % 20 == 0:
            #     print('save',i)
            #     filename_m = f"car{time.strftime('%m-%d-%H-%M-%S',time.localtime(time.time()))}_{i}.jpg"
            #
            #     cv2.imwrite(f'output/{filename_m}', frame)
                # 显示在右边对应框
                # self.MsgTE.setText(f"{i}")
                # self.MsgTE.show()
                # self.display1(frame)
                # qimg = qimage2ndarray.array2qimage(frame)
                # self.img_one.setPixmap(QPixmap(qimg).scaled(self.img_one.width(), self.img_one.height()))
                # self.img_one.show()
# 从此注释
        # start_time = time.time()
        # print("车辆检测--")
        # car_det_res , is_car= ycd.yolo_car_det("output")
        # now_time = time.time()
        # all_time = datetime.timedelta(seconds=now_time - start_time)
        # print("结束-time:", all_time)
# 除了显示# 上面已成功运行


        # if is_car:
        #     car_jpg_res = QPixmap('./yolov3_tiny_car_det/output/{}'.format(car_det_res)).scaled(self.car_det.width(),self.car_det.height())
        #     self.car_det.setPixmap(car_jpg_res)
        #     car_det_res = f'./yolov3_tiny_car_det/output/{car_det_res}'
        # else:
        #     self.car_det.setPixmap(jpg)
        # print("车牌定位--")
        # 重点关注下面传输图片和目录的区别
        # pr_det_res = pll.location_main(imgName)
        # # pr_det_res = None
        # print("车牌识别--")
        # if pr_det_res:
        #     pr_rec_res = plr.inference()
        #     now_time = time.time()
        #     all_time = datetime.timedelta(seconds=now_time - start_time)
        #     print("结束-time:",all_time)
        #     print("car:",car_det_res)
        #     print("pr：",pr_det_res)
        #
        #     jpg_res = QPixmap('{}'.format(pr_det_res)).scaled(self.pr_loc.width(), self.pr_loc.height())
        #     self.pr_loc.setPixmap(jpg_res)
        #     self.pr_res.setText(pr_rec_res)
        #     jilu = time.strftime("%b %d %Y %H:%M:%S",time.localtime(time.time())) +  '车牌号：' + pr_rec_res
        #     self.result_form.appendPlainText(jilu)
        #     self.MsgTE.setText("单张图片正常识别！")
        #
        # else:
        #     self.MsgTE.setText("单张图片未检测到车牌！")
        #     self.pr_loc.setText('no plate!!!')
        #     self.pr_res.setText('no plate!!!')

    def choose_img(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        print("单张图片位置：：",imgName)
        jpg = QPixmap(imgName).scaled(self.img_one.width(), self.img_one.height())
        self.img_one.setPixmap(jpg)
        start_time = time.time()
        print("车辆检测--")
        car_det_res , is_car= ycd.yolo_car_det(imgName)
        print('ok')
        if is_car:
            car_jpg_res = QPixmap('./yolov3_tiny_car_det/output/{}'.format(car_det_res)).scaled(self.car_det.width(),self.car_det.height())
            self.car_det.setPixmap(car_jpg_res)
            car_det_res = f'./yolov3_tiny_car_det/output/{car_det_res}'
        else:
            self.car_det.setPixmap(jpg)
        print("车牌定位--")
        # start_time = time.time()

        pr_det_res = pll.location_main(imgName)
        # pr_det_res = None
        print("车牌识别--")
        if pr_det_res:
            pr_rec_res = plr.inference()
            now_time = time.time()
            all_time = datetime.timedelta(seconds=now_time - start_time)
            print("结束-time:",all_time)
            print("car:",car_det_res)
            print("pr：",pr_det_res)

            jpg_res = QPixmap('{}'.format(pr_det_res)).scaled(self.pr_loc.width(), self.pr_loc.height())
            self.pr_loc.setPixmap(jpg_res)
            self.pr_res.setText(pr_rec_res)
            jilu = time.strftime("%b %d %Y %H:%M:%S",time.localtime(time.time())) +  '车牌号：' + pr_rec_res
            self.result_form.appendPlainText(jilu)
            self.MsgTE.setText("单张图片正常识别！")

        else:
            self.MsgTE.setText("单张图片未检测到车牌！")
            self.pr_loc.setText('no plate!!!')
            self.pr_res.setText('no plate!!!')

    def SetR(self):
        R=self.RedColorSld.value()
        self.R=R/255
    def SetG(self):
        G=self.GreenColorSld.value()
        self.G=G/255
    def SetB(self):
        B=self.BlueColorSld.value()
        self.B=B/255
    def SetContrast(self):
        contrast_toset=self.ContrastSld.value()
        try:
            self.camera.set(11,contrast_toset)
            self.MsgTE.setPlainText('The contrast is set to ' + str(self.camera.get(11)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))
    def SetBrightness(self):
        brightness_toset=self.BrightSld.value()
        try:
            self.camera.set(10,brightness_toset)
            self.MsgTE.setPlainText('The brightness is set to ' + str(self.camera.get(10)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))
    def SetGain(self):
        gain_toset=self.GainSld.value()
        try:
            self.camera.set(14,gain_toset)
            self.MsgTE.setPlainText('The gain is set to '+str(self.camera.get(14)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))
    def SetExposure(self):
        try:
            exposure_time_toset=self.ExpTimeSld.value()
            self.camera.set(15,exposure_time_toset)
            self.MsgTE.setPlainText('The exposure time is set to '+str(self.camera.get(15)))
        except Exception as e:
            self.MsgTE.setPlainText(str(e))
    def SetGray(self):
        if self.GrayImgCkB.isChecked():
            self.RedColorSld.setEnabled(False)
            self.RedColorSpB.setEnabled(False)
            self.GreenColorSld.setEnabled(False)
            self.GreenColorSpB.setEnabled(False)
            self.BlueColorSld.setEnabled(False)
            self.BlueColorSpB.setEnabled(False)
        else:
            self.RedColorSld.setEnabled(True)
            self.RedColorSpB.setEnabled(True)
            self.GreenColorSld.setEnabled(True)
            self.GreenColorSpB.setEnabled(True)
            self.BlueColorSld.setEnabled(True)
            self.BlueColorSpB.setEnabled(True)
    def StartCamera(self):
        self.ShowBt.setEnabled(False)
        self.StopBt.setEnabled(True)
        self.RecordBt.setEnabled(True)
        self.GrayImgCkB.setEnabled(True)
        if self.GrayImgCkB.isChecked()==0:
            self.RedColorSld.setEnabled(True)
            self.RedColorSpB.setEnabled(True)
            self.GreenColorSld.setEnabled(True)
            self.GreenColorSpB.setEnabled(True)
            self.BlueColorSld.setEnabled(True)
            self.BlueColorSpB.setEnabled(True)
        self.ExpTimeSld.setEnabled(True)
        self.ExpTimeSpB.setEnabled(True)
        self.GainSld.setEnabled(True)
        self.GainSpB.setEnabled(True)
        self.BrightSld.setEnabled(True)
        self.BrightSpB.setEnabled(True)
        self.ContrastSld.setEnabled(True)
        self.ContrastSpB.setEnabled(True)
        self.RecordBt.setText('录像')

        self.Timer.start(1)
        self.timelb=time.clock()
    def SetFilePath(self):
        dirname = QFileDialog.getExistingDirectory(self, "浏览", '.')
        if dirname:
            self.FilePathLE.setText(dirname)
            self.RecordPath=dirname+'/'
    def TimerOutFun(self):
        success,img=self.camera.read()
        if success:
            self.Image = self.ColorAdjust(img)
            #self.Image=img
            self.DispImg()
            self.Image_num+=1
            if self.RecordFlag:
                self.video_writer.write(img)
            if self.Image_num%10==9:
                frame_rate=10/(time.clock()-self.timelb)
                self.FmRateLCD.display(frame_rate)
                self.timelb=time.clock()
                #size=img.shape
                self.ImgWidthLCD.display(self.camera.get(3))
                self.ImgHeightLCD.display(self.camera.get(4))
        else:
            self.MsgTE.clear()
            self.MsgTE.setPlainText('Image obtaining failed.')
    def ColorAdjust(self,img):
        try:
            B=img[:,:,0]
            G=img[:,:,1]
            R=img[:,:,2]
            B=B*self.B
            G=G*self.G
            R=R*self.R
            #B.astype(cv2.PARAM_UNSIGNED_INT)
            #G.astype(cv2.PARAM_UNSIGNED_INT)
            #R.astype(cv2.PARAM_UNSIGNED_INT)

            img1=img
            img1[:,:,0]=B
            img1[:,:,1]=G
            img1[:,:,2]=R
            return img1
        except Exception as e:
            self.MsgTE.setPlainText(str(e))
    def DispImg(self):
        if self.GrayImgCkB.isChecked():
            img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
        qimg = qimage2ndarray.array2qimage(img)
        self.DispLb.setPixmap(QPixmap(qimg))
        self.DispLb.show()
    def StopCamera(self):
        if self.StopBt.text()=='暂停':
            self.StopBt.setText('继续')
            self.RecordBt.setText('保存')
            self.Timer.stop()
        elif self.StopBt.text()=='继续':
            self.StopBt.setText('暂停')
            self.RecordBt.setText('录像')
            self.Timer.start(1)
    def RecordCamera(self):
        tag=self.RecordBt.text()
        if tag=='保存':
            try:
                image_name=self.RecordPath+'image'+time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))+'.jpg'
                print(image_name)
                cv2.imwrite(image_name, self.Image)
                # jpg = QPixmap(image_name).scaled(self.show_img.width(), self.show_img.height())
                # print(jpg)
                # self.show_img.setPixmap(jpg)
                self.MsgTE.clear()
                self.MsgTE.setPlainText('Image saved.{}'.format(image_name))
            except Exception as e:
                self.MsgTE.clear()
                self.MsgTE.setPlainText(str(e))
        elif tag=='录像':
            self.RecordBt.setText('停止')

            video_name = self.RecordPath + 'video' + time.strftime('%Y%m%d%H%M%S',time.localtime(time.time())) + '.avi'
            fps = self.FmRateLCD.value()
            size = (self.Image.shape[1],self.Image.shape[0])
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            self.video_writer = cv2.VideoWriter(video_name, fourcc,self.camera.get(5), size)
            self.RecordFlag=1
            self.MsgTE.setPlainText('Video recording...')
            self.StopBt.setEnabled(False)
            # self.ExitBt.setEnabled(False)
        elif tag == '停止':
            self.RecordBt.setText('录像')
            self.video_writer.release()
            self.RecordFlag = 0
            self.MsgTE.setPlainText('Video saved.')
            self.StopBt.setEnabled(True)
            # self.ExitBt.setEnabled(True)
    def ExitApp(self):
        self.Timer.Stop()
        self.camera.release()
        self.MsgTE.setPlainText('Exiting the application..')
        QCoreApplication.quit()

    # def choose_pat(self):




































if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui=CamShow()
    ui.show()
    sys.exit(app.exec_())