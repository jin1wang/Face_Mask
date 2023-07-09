
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp


# 初始化一个旋转矩阵的列表
rot_matrices = []
keypoints_history = []
max_rot_length = 5
max_history_length = 10

# 加载面具图像
mask_img = cv2.cvtColor(cv2.imread('image/facepaint.png', cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2BGRA)
df = pd.read_csv('image/labels_facepaint.csv', header=None)
coordinates = df.values
coordinates = df[[1, 2]].values

# 初始化MediaPipe人脸检测模型
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5)

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取图像
    ret, frame = cap.read()
    height, width, _ = frame.shape

    # 将图像转换为RGB格式，并输入给MediaPipe模型
    results = mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 创建一个空白遮罩图像
    mask = np.zeros_like(frame, dtype=np.uint8)

    # 检测人脸关键点
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark

        face_points = []
        for landmark in face_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            face_points.append([x, y])
        face_points_np = np.array(face_points) 

        # 提取眼睛和嘴巴区域的关键点索引
        eye_indices_l = [130, 247, 30, 29, 27, 28, 56, 190, 243, 112, 26, 22, 23, 24, 110, 25]   # 16
        eye_indices_r = [463, 414, 286, 258, 257, 259, 260, 467, 359, 255, 339, 254, 253, 252, 256, 341]   # 16
        mouth_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]   # 20
        fc_indices = [234, 162, 54, 67, 10, 297, 284, 389, 454, 361, 397, 379, 400, 152, 176, 150, 172, 132]  # 18
        total_indices = eye_indices_l + eye_indices_r + mouth_indices + fc_indices
        # 提取眼睛和嘴巴区域的关键点
        eye_landmarks_l = [(int(face_landmarks[i].x * width), int(face_landmarks[i].y * height)) for i in eye_indices_l]
        eye_landmarks_r = [(int(face_landmarks[i].x * width), int(face_landmarks[i].y * height)) for i in eye_indices_r]
        mouth_landmarks = [(int(face_landmarks[i].x * width), int(face_landmarks[i].y * height)) for i in mouth_indices]
        total_landmarks = [(int(face_landmarks[i].x * width), int(face_landmarks[i].y * height)) for i in total_indices]
        # total_landmarks = [(int(face_landmarks[i].x * width), int(face_landmarks[i].y * height), int(face_landmarks[i].z * 1000)) for i in total_indices

        keypoints_history.append(total_landmarks)
        if len(keypoints_history) > max_history_length:
            keypoints_history.pop(0)
        smoothed_landmarks = np.mean(keypoints_history, axis=0).astype(np.int32)

        # 创建用于绘制多边形的点的列表
        eye_points_l = np.array(eye_landmarks_l, np.int32)
        eye_points_r = np.array(eye_landmarks_r, np.int32)
        mouth_points = np.array(mouth_landmarks, np.int32)
        total_points = np.array(total_landmarks, np.int32)

        matrix, _ = cv2.findHomography(coordinates, total_points)

        rot_matrices.append(matrix)
        # 如果列表中的旋转矩阵数量超过N，就删除最旧的旋转矩阵
        if len(rot_matrices) > max_rot_length:
            rot_matrices.pop(0)
        avg_matrix = np.mean(rot_matrices, axis=0)
        transformed_img = cv2.warpPerspective(mask_img, matrix, (frame.shape[1], frame.shape[0]))

        # 使用凸包算法对关键点进行排序
        hull = cv2.convexHull(face_points_np.astype(np.int32))  # 返回凸包的顶点
        eye_hull_l = cv2.convexHull(eye_points_l.astype(np.int32))
        eye_hull_r = cv2.convexHull(eye_points_r.astype(np.int32))
        mouth_hull = cv2.convexHull(mouth_points.astype(np.int32))
        # 在遮罩图像上绘制人脸区域
        cv2.fillPoly(mask, [hull.astype(np.int32)], (255, 255, 255), cv2.LINE_AA)
        cv2.fillPoly(mask, [eye_hull_l.astype(np.int32)], (0, 0, 0, 255), cv2.LINE_AA)
        cv2.fillPoly(mask, [eye_hull_r.astype(np.int32)], (0, 0, 0, 255), cv2.LINE_AA)
        cv2.fillPoly(mask, [mouth_hull.astype(np.int32)], (0, 0, 0, 255), cv2.LINE_AA)

        # 将变换后的源图像贴到目标图像上
        transformed_img = cv2.cvtColor(transformed_img, cv2.COLOR_BGRA2BGR)
        result = cv2.bitwise_and(transformed_img, mask)
        result = cv2.add(result, cv2.bitwise_and(frame, 255 - mask))

        # 显示结果图像
        cv2.imshow('Result', result)

    # 退出程序
    if cv2.waitKey(1) == ord('q'):
        break

# 关闭摄像头
cap.release()
cv2.destroyAllWindows()