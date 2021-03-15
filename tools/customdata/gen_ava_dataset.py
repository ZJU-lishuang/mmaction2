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

if __name__ == '__main__':
    dataset_xml='/home/lishuang/Disk/dukto/default09'
    video_save_path='rawframes/'
    save_image=False
    _FPS = 25

    xml_names= loadAllTagFile(dataset_xml, '.xml')
    xml_names.sort()
    exist_image=False
    #标注的关键帧图片是否存在，用于检查是否有漏标，不是必须的
    if exist_image:
        dataset_folder = '/home/lishuang/Disk/dukto/异常行为现场数据/问讯室'
        fig_names = loadAllTagFile(dataset_folder, '.jpg')
        fig_names.sort()
    else:
        fig_names=xml_names

    total_ann={}
    total_entity_id=[]
    total_action_labels=[]
    for fig_name in fig_names:
        img_basename=os.path.basename(fig_name)
        xml_basename=os.path.splitext(img_basename)[0]+".xml"
        xml_path=os.path.join(dataset_xml,xml_basename)
        assert xml_path in xml_names
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('imagesize')
        w = int(size.find('ncols').text)
        h = int(size.find('nrows').text)
        video_id,frame_id=os.path.splitext(img_basename)[0].rsplit('_', 1)
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
        assert timestamp not in total_ann[video_id]
        total_ann[video_id][timestamp]={}
        person_id=0
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name != '异常行为人':
                continue
            person_id+=1
            total_ann[video_id][timestamp][person_id] = {}
            labels = obj.find('attributes').text
            # pattern = r"异常行为="
            # m = re.search(pattern, labels)
            action_label,entity_id=labels.split(',')
            polygon = obj.find('polygon')
            pts = polygon.findall('pt')
            bbox = [
                float(pts[0].find('x').text)/w,
                float(pts[0].find('y').text)/h,
                float(pts[2].find('x').text)/w,
                float(pts[2].find('y').text)/h
            ]
            total_ann[video_id][timestamp][person_id]['bbox']=bbox
            total_ann[video_id][timestamp][person_id]['label']=action_label
            total_ann[video_id][timestamp][person_id]['id']=entity_id
            if action_label not in total_action_labels:
                total_action_labels.append(action_label)
            if entity_id not in total_entity_id:
                total_entity_id.append(entity_id)

    entity_ids = {name: i for i, name in enumerate(total_entity_id)}
    label_ids = {name: i for i, name in enumerate(total_action_labels)}
    file_data = ""
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

                # proposals[img_key].append(np.array([bbox[0],bbox[1],bbox[2],bbox[3],percent]))
                tmp_proposals.append([bbox[0], bbox[1], bbox[2], bbox[3], percent])
            proposals[img_key] = np.array(tmp_proposals)

    mmcv.dump(proposals,'ava_customdataset_proposals_train.pkl')

    with open('ava_train_customdataset.csv', 'w') as f:
        f.write(file_data)