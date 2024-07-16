import cv2
import os

# 定义图片目录和视频输出路径
# data_folder = 'data'
# output_file = '0.mp4'
data_folder = 'data_after_attacking'
output_file = '1.mp4'
# frame_size = (1088, 608)
frame_size = (640, 360)
# frame_size = (640, 480)
fps = 25  # 每秒帧数

# 获取文件夹中所有文件，按文件名排序
files = sorted(os.listdir(data_folder))

# 过滤符合条件的文件，并限制数量为400
images = [file for file in files if file.endswith('.jpg') and file[:-4].isdigit() and int(file[:-4]) <= 400]
images = images[:400]
print(f"共找到 {len(images)} 张图片。")

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

image_index = 0
# 遍历图片并写入视频
for image in images:
    image_index += 1
    # print(f"正在处理第 {image_index} 张图片...")
    img_path = os.path.join(data_folder, image)
    img = cv2.imread(img_path)
    if img is not None:
        # resized_img = cv2.resize(img, frame_size)
        video_writer.write(img)
        
video_size = os.path.getsize(output_file)

# 打印视频信息
print("视频信息：")
print("帧率：", fps)
print("分辨率：", frame_size)
print("大小：", video_size, "字节")

# 释放视频写入对象
video_writer.release()
print("视频已保存到", output_file)
