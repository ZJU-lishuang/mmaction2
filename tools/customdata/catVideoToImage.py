# encoding:utf-8
import cv2
import os


def isfile(src_path):
    if os.path.isfile(src_path):
        file_path = os.path.dirname(src_path)  # 去掉文件名，返回所在目录

        cat_video(file_path)
        return True
    else:
        infile(src_path)
        return False


def infile(src_path):
    for item in os.listdir(src_path):
        file_path = os.path.join(src_path, item)
        paduan = isfile(file_path)
        if paduan:
            break


def cat_video(file_path):
    for videofile in os.listdir(file_path):
        print(videofile)
        imagefile_name = videofile.split(".")[0]
        imagefile_path = os.path.join(video_dst_path, imagefile_name)
        if not os.path.exists(imagefile_path):
            os.mkdir(imagefile_path)
        videofile_path = os.path.join(file_path, videofile)
        videoName = videofile_path.split("/")[-1].split(".")[0]
        image_name = ""
        if nameLength != 0 or image_name != "":
            image_name = str(videoName)[0:nameLength]
        else:
            image_name = str(videoName)
        cap = cv2.VideoCapture(videofile_path)
        if cap.isOpened():
            rval, frame = cap.read()
        else:
            rval = False
        count = 1
        while rval:
            rval, frame = cap.read()
            imageName=f'img_{count:05d}.jpg'
            if count % stop == 0:
                try:
                    cv2.imwrite(os.path.join(imagefile_path, imageName), frame)
                except:
                    continue
            count += 1
            print(count)

        cap.release


if __name__ == "__main__":
    video_path = "/home/lishuang/Disk/dukto/异常行为采集"  # 视频路径
    video_dst_path = "/home/lishuang/Disk/dukto/异常行为采集图片"  # 图片保存路径
    if not os.path.exists(video_dst_path):
        os.mkdir(video_dst_path)
    nameLength = 0  # 图片名长度  如果长度为0则使用视频原名
    stop = 1  # 视频截取帧数间隔
    isfile(video_path)
