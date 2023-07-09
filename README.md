# AR_Face_Mask

#### 基于Mediapipe的468人脸关键点检测，使用opencv进行人脸面具的佩戴。

#### 安装环境

1.opencv  
2.mediapipe  
3.padas

    创建Conda环境，使用以下命令安装所使用的包：  
    pip install python-opencv -i https://pypi.tuna.tsinghua.edu.cn/simple  
    pip install mediapipe -i https://pypi.tuna.tsinghua.edu.cn/simple  
    pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple

#### 使用说明

1. 绘制面具以及标注关键点

    首先使用标准脸模型绘制自己想要的图案：  
    ![image](https://github.com/jin1wang/Face_Mask/tree/main/image/doc/468key_landmark.png)  
    注：绘制的图片上下左右边尽量不要留白。

    其次标注面具关键点，本文使用 https://www.makesense.ai/ 在线工具标注关键点，标注完成后保存生成csv文件：
    ![image](https://github.com/jin1wang/Face_Mask/tree/main/image/doc/Annotation.jpg)   

3. 注意在mask.py里将图片以及csv文件改成自己的路径，运行:python mask.py
4. 效果图如下:  
    ![image](https://github.com/jin1wang/Face_Mask/tree/main/image/doc/result.gif)  
