import os
import xml.etree.ElementTree as ET
import numpy as np
import mmcv
import shutil
# import re

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

if __name__ == '__main__':
    #标注文件夹
    # dataset_xml='/home/lishuang/Disk/dukto/action_train'
    # proposals_pkl='ava_customdataset_proposals_train.pkl'
    # csv_file='ava_train_customdataset.csv'
    dataset_xml = '/home/lishuang/Disk/gitlab/traincode/video_action_det/data/ava_xianchang/'
    proposals_pkl = 'ava_customdataset_proposals_train.pkl'
    csv_file = 'ava_train_customdataset.csv'

    dataset_xml1 = '/home/lishuang/Disk/dukto/异常行为标注/action_train'
    
    #训练集10帧抽取，验证集25帧抽取，取公约数5
    _FPS = 5
    #获取文件夹下对应后缀的文件，包括子文件夹
    xml_names= GetFileFromThisRootDir(dataset_xml, '.xml')
    xml_names.sort()

    xml_names1 = GetFileFromThisRootDir(dataset_xml1, '.xml')
    xml_names1.sort()

    xml_names=xml_names+xml_names1

    exist_image=False
    save_image=False
    #标注的关键帧图片是否存在，用于检查是否有漏标，不是必须的
    if exist_image:
        #图片路径
        dataset_folder = ''
        fig_names = loadAllTagFile(dataset_folder, '.jpg')
        fig_names.sort()
        #是否将标注图片按视频分开保存
        if save_image:
            video_save_path = 'rawframes/'
    else:
        fig_names=xml_names
        save_image=False

    total_ann={}
    total_entity_id=[]
    total_action_labels=[]
    total_action_labels_num = {}
    for fig_name in fig_names:
        img_basename=os.path.basename(fig_name)
        xml_basename=os.path.splitext(img_basename)[0]+".xml"
        xml_path=os.path.join(os.path.dirname(fig_name),xml_basename)
        assert xml_path in xml_names
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('imagesize')
        w = int(size.find('ncols').text)
        h = int(size.find('nrows').text)
        video_id,frame_id=os.path.splitext(img_basename)[0].rsplit('_', 1)
        #异常处理，有的是 frame_图片帧.xml，有的是 视频名称_图片帧.xml :(
        if video_id == 'frame':
            video_id=os.path.basename(os.path.dirname(xml_path))
            if video_id =="default":
                video_id = os.path.basename(os.path.dirname(os.path.dirname(xml_path)))
        if video_id not in total_ann:
            total_ann[video_id]={}
            if save_image:
                new_dir = os.path.join(video_save_path, video_id)
                if not os.path.isdir(new_dir):
                    print(f'Creating folder: {new_dir}')
                    os.makedirs(new_dir)
        #save image
        if save_image:
            save_image_path=os.path.join(video_save_path, video_id,img_basename)
            shutil.copy(fig_name, save_image_path)
        timestamp=int(int(frame_id)/_FPS)
        assert timestamp not in total_ann[video_id],f'video_id={video_id},timestamp={timestamp},frame_id={frame_id},fps={_FPS}'
        total_ann[video_id][timestamp]={}
        person_id=0
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name != '异常行为人' and name != '正常行为人':
                continue
            person_id+=1
            total_ann[video_id][timestamp][person_id] = {}
            labels = obj.find('attributes').text
            # pattern = r"异常行为="
            # m = re.search(pattern, labels)

            #异常处理：用视频标注的内容，里面属性不一致
            # if len(labels.split(',')) ==4:
            #     _,action_label,_, entity_id = labels.split(',')
            #     if '人物ID' not in entity_id:
            #         entity_id,action_label,_, _ = labels.split(',')
            # else:
            #     assert len(labels.split(',')) ==2
            #     entity_id, action_label = labels.split(',')
            #     if '异常行为' not in action_label:
            #         action_label, entity_id = labels.split(',')
            for label in labels.split(','):
                if '人物ID' in label:
                    entity_id = label
                if '异常行为' in label:
                    action_label = label
            if name == "正常行为人":
                action_label="异常行为=正常"
                for label in labels.split(','):
                    if 'track_id' in label:
                        entity_id = video_id+'_'+label

            #验证集，历史遗留，先采样间隔25帧，后改为间隔10帧，历史遗留问题，先用图片标注，后用视频标注导致不同
            # 异常处理：标注文件问题，标注信息可能会左右对调
            # entity_id, action_label = labels.split(',')
            # if '异常行为' not in action_label:
            #     action_label, entity_id = labels.split(',')

            # assert '异常行为' in action_label,f'xml_path={xml_path}'
            # assert '人物ID' in entity_id,f'xml_path={xml_path}'

            polygon = obj.find('polygon')
            pts = polygon.findall('pt')
            bbox = [
                float(pts[0].find('x').text)/w,
                float(pts[0].find('y').text)/h,
                float(pts[2].find('x').text)/w,
                float(pts[2].find('y').text)/h
            ]
            action_label=action_label.strip()
            entity_id=entity_id.strip()
            total_ann[video_id][timestamp][person_id]['bbox']=bbox
            total_ann[video_id][timestamp][person_id]['label']=action_label
            total_ann[video_id][timestamp][person_id]['id']=entity_id
            if action_label not in total_action_labels:
                total_action_labels.append(action_label)
            if entity_id not in total_entity_id:
                total_entity_id.append(entity_id)
            if action_label not in total_action_labels_num:
                total_action_labels_num[action_label]=1
            else:
                total_action_labels_num[action_label]+=1


    def custom_action_labels():
        return [
            '异常行为=头撞墙', '异常行为=砸门', '异常行为=正常', '异常行为=扇巴掌', '异常行为=掐脖子', '异常行为=举高', '异常行为=撞桌', '异常行为=打斗',
            '异常行为=打滚', '异常行为=快速移动', '异常行为=举标语', '异常行为=发传单'
        ]

    entity_ids = {name: i for i, name in enumerate(total_entity_id)}
    label_ids = {name: i for i, name in enumerate(custom_action_labels())}
    file_data = ""
    print(total_action_labels)
    print(total_action_labels_num)
    print(label_ids)
    proposals={}
    for video_name,total_frames in total_ann.items():
        for frame,frame_ann in total_frames.items():
            if len(frame_ann) == 0:
                continue
            img_key = f'{video_name},{frame:04d}'
            percent = 0.95
            tmp_proposals=[]
            for person_id,person_ann in frame_ann.items():

                bbox=person_ann['bbox']
                label_id = label_ids[person_ann['label']]+1
                entity_id=entity_ids[person_ann['id']]
                file_data+= f"{video_name},{frame},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{label_id},{entity_id}\n"

                tmp_proposals.append([bbox[0], bbox[1], bbox[2], bbox[3], percent])
            proposals[img_key] = np.array(tmp_proposals)

    mmcv.dump(proposals,proposals_pkl)

    with open(csv_file, 'w') as f:
        f.write(file_data)