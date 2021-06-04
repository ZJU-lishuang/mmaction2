import cv2
import os
import numpy as np


def subfiles(path):
    """Yield directory names not starting with '.' under given path."""
    for entry in os.scandir(path):
        if not entry.name.startswith('.') and not entry.is_dir():
            yield entry.name

def loadAllTagFile( DirectoryPath, tag ):# download all files' name
    result = []
    for file in subfiles(DirectoryPath):
    # for file in os.listdir(DirectoryPath):
        file_path = os.path.join(DirectoryPath, file)
        if os.path.splitext(file_path)[1] == tag:
            result.append(file_path)
    return result


def video2image(videoname,savepath):
    videobasename=os.path.basename(videoname)
    filebasename,extension=os.path.splitext(videobasename)
    dirname=os.path.join(savepath,filebasename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    cap=cv2.VideoCapture(videoname)
    rval=cap.isOpened()
    frame_count = 0
    while rval:
        frame_count += 1
        rval, frame = cap.read()
        if rval:
            # imgmask=np.zeros([480,640],dtype=np.uint8) #mask
            # imgmask[125:413,39:545]=255
            # image=cv2.add(frame, np.zeros(np.shape(frame), dtype=np.uint8), mask=imgmask)

            # if frame_count%10 !=0:
            #     continue

            # image=frame[125:413,39:545]  #裁剪
            image=frame.copy()

            # image[:,:]=255
            # image[125:413,39:545]=frame[125:413,39:545]

            cv2.imwrite('{}/{}.jpg'.format(dirname,frame_count), image)


    cap.release()


videopath="/home/lishuang/Disk/gitlab/traincode/mmaction2/demo/test"
savepath="/home/lishuang/Disk/gitlab/traincode/mmaction2/demo/tmp"

fig_names = loadAllTagFile(videopath, '.mp4')


for i in range(len(fig_names)):
    videoname=fig_names[i]
    video2image(videoname,savepath)

