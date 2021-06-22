import os
import xml.etree.ElementTree as ET
import numpy as np
import mmcv
import shutil
# import re
import glob

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

def GetFileFromThisRootDir(dir,ext = None):
  allfiles = []
  needExtFilter = (ext != None)
  for root,dirs,files in os.walk(dir):
    for filespath in files:
      filepath = os.path.join(root, filespath)
      extension = os.path.splitext(filepath)[1]
      if needExtFilter and extension in ext:
        allfiles.append(filepath)
      elif not needExtFilter:
        allfiles.append(filepath)
  return allfiles

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == '__main__':
    #标注文件夹
    # dataset_xml='/home/lishuang/Disk/dukto/action_train'
    # proposals_pkl='ava_customdataset_proposals_train.pkl'
    # csv_file='ava_train_customdataset.csv'
    # dataset_xml = '/home/lishuang/Disk/gitlab/traincode/video_action_det/data/ava_xianchang/天府机场标签'
    # proposals_pkl = 'ava_customdataset_proposals_train.pkl'
    # csv_file = 'ava_train_customdataset.csv'

    dataset_xml = '/home/lishuang/Disk/dukto/123tmp/action_val'
    
    #训练集10帧抽取，验证集25帧抽取，取公约数5
    _FPS = 5
    #获取文件夹下对应后缀的文件，包括子文件夹
    xml_names= GetFileFromThisRootDir(dataset_xml, '.xml')
    xml_names.sort()

    dataset_folder = '/home/lishuang/Disk/gitlab/traincode/video_action_det/data/ava_custom/*/*/*.jpg'
    fig_names = glob.glob(dataset_folder)
    fig_names.sort()


    frame_info={}
    total_video=[]
    for xml_name in xml_names:
        xml_basename = os.path.basename(xml_name)
        video_id, frame_id = os.path.splitext(xml_basename)[0].rsplit('_', 1)
        if video_id == 'frame':
            video_id=os.path.basename(os.path.dirname(xml_name))
            if video_id =="default":
                video_id = os.path.basename(os.path.dirname(os.path.dirname(xml_name)))
        if video_id not in frame_info:
            frame_info[video_id]={}
        #标注从0开始，视频解帧的图片从1开始,对齐frame_id
        frame_id=f'{(int(frame_id)+1):05d}'
        if frame_id not in frame_info[video_id]:
            frame_info[video_id][frame_id]= {}
            frame_info[video_id][frame_id]["ann_path"] = xml_name
    for fig_name in fig_names:
        img_basename=os.path.basename(fig_name)
        video_id=os.path.basename(os.path.dirname(fig_name))
        _,frame_id=os.path.splitext(img_basename)[0].rsplit('_', 1)
        if video_id in frame_info and frame_id in frame_info[video_id]:
            frame_info[video_id][frame_id]["img_path"] = fig_name
    save_img=True
    video_save_path = './images/'
    for video_name,total_frames in frame_info.items():
        txtdir = f"./labels/{video_name}"
        check_dir(os.path.join(video_save_path, video_name))
        for frame_id,frame_info in total_frames.items():
            img_path = frame_info["img_path"]
            ann_path = frame_info["ann_path"]
            tree = ET.parse(ann_path)
            root = tree.getroot()
            size = root.find('imagesize')
            width = int(size.find('ncols').text)
            height = int(size.find('nrows').text)
            lines=[]
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name != '异常行为人' and name != '正常行为人':
                    continue
                labels = obj.find('attributes').text
                polygon = obj.find('polygon')
                pts = polygon.findall('pt')

                x=float(pts[0].find('x').text)
                y=float(pts[0].find('y').text)
                x2=float(pts[2].find('x').text)
                y2=float(pts[2].find('y').text)
                cx = (x2 + x) * 0.5 / width
                cy = (y2 + y) * 0.5 / height
                w = (x2 - x) * 1. / width
                h = (y2 - y) * 1. / height
                label="0"
                line = "%s %.6f %.6f %.6f %.6f\n" % (label, cx, cy, w, h)
                lines.append(line)
            if len(lines) > 0:
                txtbasename = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
                txt_name = os.path.join(txtdir, txtbasename)
                check_dir(os.path.dirname(txt_name))
                with open(txt_name, "w") as f:
                    f.writelines(lines)
                if save_img:
                    save_image_path=os.path.join(video_save_path, video_name,os.path.basename(img_path))
                    shutil.copy(img_path, save_image_path)