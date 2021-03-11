import os
import xml.etree.ElementTree as ET
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
    dataset_folder='/home/lishuang/Disk/dukto/异常行为现场数据/问讯室'
    dataset_xml='/home/lishuang/Disk/dukto/default09'

    fig_names = loadAllTagFile(dataset_folder, '.jpg')
    xml_names= loadAllTagFile(dataset_xml, '.xml')
    xml_names.sort()
    fig_names.sort()

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
        video_id,timestamp=os.path.splitext(img_basename)[0].rsplit('_', 1)
        if video_id not in total_ann:
            total_ann[video_id]={}
        assert timestamp not in total_ann[video_id]
        total_ann[video_id][timestamp]={}
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name != '异常行为人':
                continue
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
            total_ann[video_id][timestamp]['bbox']=bbox
            total_ann[video_id][timestamp]['label']=action_label
            total_ann[video_id][timestamp]['id']=entity_id
            if action_label not in total_action_labels:
                total_action_labels.append(action_label)
            if entity_id not in total_entity_id:
                total_entity_id.append(entity_id)

    entity_ids = {name: i for i, name in enumerate(total_entity_id)}
    label_ids = {name: i for i, name in enumerate(total_action_labels)}
    file_data = ""
    print(label_ids)
    for video_name,total_frames in total_ann.items():
        for frame,frame_ann in total_frames.items():
            if len(frame_ann) == 0:
                continue
            bbox=frame_ann['bbox']
            label_id = label_ids[frame_ann['label']]
            entity_id=entity_ids[frame_ann['id']]
            file_data+= f"{video_name},{frame},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{label_id},{entity_id}\n"


    with open('ava_train_customdataset.csv', 'w') as f:
        f.write(file_data)