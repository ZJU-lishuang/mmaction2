import argparse
import copy as cp
import os
import os.path as osp
import shutil

import cv2
import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from tqdm import tqdm


from xml.etree import ElementTree as ET

from mmaction.models import build_detector
from mmaction.utils import import_module_error_func

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):

    @import_module_error_func('mmdet')
    def inference_detector(*args, **kwargs):
        pass

    @import_module_error_func('mmdet')
    def init_detector(*args, **kwargs):
        pass


try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 1
LINETYPE = 1


#创建一级分支object
def create_object(root,label,xi,yi,xa,ya):#参数依次，树根，xmin，ymin，xmax，ymax
    #创建一级分支object
    _object=ET.SubElement(root,'object')
    #创建二级分支
    name=ET.SubElement(_object,'name')
    name.text='异常行为人'
    # name.text = str(label)
    pose=ET.SubElement(_object,'deleted')
    pose.text='0'
    truncated=ET.SubElement(_object,'verified')
    truncated.text='0'
    difficult=ET.SubElement(_object,'occluded')
    difficult.text='no'
    date = ET.SubElement(_object, 'date')
    date.text = ''
    id = ET.SubElement(_object, 'id')
    id.text = '-1'
    # 创建一级分支parts
    parts = ET.SubElement(_object, 'parts')
    # 创建source下的二级分支hasparts
    hasparts = ET.SubElement(parts, 'hasparts')
    hasparts.text = ''
    # 创建source下的二级分支ispartof
    ispartof = ET.SubElement(parts, 'ispartof')
    ispartof.text = ''
    type = ET.SubElement(_object, 'type')
    type.text = 'bounding_box'
    #创建bndbox
    bndbox=ET.SubElement(_object,'polygon')
    pt1 = ET.SubElement(bndbox, 'pt')
    x1 = ET.SubElement(pt1, 'x')
    x1.text = '%s'%xi
    y1 = ET.SubElement(pt1, 'y')
    y1.text = '%s'%yi
    pt2 = ET.SubElement(bndbox, 'pt')
    x2 = ET.SubElement(pt2, 'x')
    x2.text = '%s' % xa
    y2 = ET.SubElement(pt2, 'y')
    y2.text = '%s' % yi
    pt3 = ET.SubElement(bndbox, 'pt')
    x3 = ET.SubElement(pt3, 'x')
    x3.text = '%s' % xa
    y3 = ET.SubElement(pt3, 'y')
    y3.text = '%s' % ya
    pt4 = ET.SubElement(bndbox, 'pt')
    x4 = ET.SubElement(pt4, 'x')
    x4.text = '%s' % xi
    y4 = ET.SubElement(pt4, 'y')
    y4.text = '%s' % ya

    username = ET.SubElement(bndbox, 'username')
    username.text = ''
    attributes = ET.SubElement(_object, 'attributes')
    attributes.text = ''
    # attributes.clear()

#创建xml文件
def create_tree(image_name):
    global annotation
    # 创建树根annotation
    annotation = ET.Element('annotation')

    #创建一级分支filename
    filename=ET.SubElement(annotation,'filename')
    filename.text=os.path.basename(image_name).strip('.jpg')

    #创建一级分支folder
    folder = ET.SubElement(annotation,'folder')
    #添加folder标签内容
    folder.text=('ls')

    # 创建一级分支source
    source = ET.SubElement(annotation, 'source')
    # 创建source下的二级分支sourceImage
    database = ET.SubElement(source, 'sourceImage')
    database.text = ''
    # 创建source下的二级分支sourceAnnotation
    database = ET.SubElement(source, 'sourceAnnotation')
    database.text = 'Datumaro'

    imgtmp = cv2.imread(image_name)
    imgheight,imgwidth,imgdepth=imgtmp.shape
    #创建一级分支size
    size=ET.SubElement(annotation,'imagesize')
    #创建size下的二级分支图像的宽、高及depth
    height = ET.SubElement(size, 'nrows')
    height.text = str(imgheight)
    width=ET.SubElement(size,'ncols')
    width.text=str(imgwidth)


def pretty_xml(element, indent, newline, level=0):  # elemnt为传进来的Elment类，参数indent用于缩进，newline用于换行
    if element:  # 判断element是否有子元素
        if (element.text is None) or element.text.isspace():  # 如果element的text没有内容
            element.text = newline + indent * (level + 1)
        else:
            element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * (level + 1)
            # else:  # 此处两行如果把注释去掉，Element的text也会另起一行
            # element.text = newline + indent * (level + 1) + element.text.strip() + newline + indent * level
    temp = list(element)  # 将element转成list
    for subelement in temp:
        if temp.index(subelement) < (len(temp) - 1):  # 如果不是list的最后一个元素，说明下一个行是同级别元素的起始，缩进应一致
            subelement.tail = newline + indent * (level + 1)
        else:  # 如果是list的最后一个元素， 说明下一行是母元素的结束，缩进应该少一个
            subelement.tail = newline + indent * level
        pretty_xml(subelement, indent, newline, level=level + 1)  # 对子元素进行递归操作



def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))


plate_blue = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
plate_blue = plate_blue.split('-')
plate_blue = [hex2color(h) for h in plate_blue]
plate_green = '004b23-006400-007200-008000-38b000-70e000'
plate_green = plate_green.split('-')
plate_green = [hex2color(h) for h in plate_green]


def visualize(frames, annotations, plate=plate_blue, max_num=5):
    """Visualize frames with predicted annotations.

    Args:
        frames (list[np.ndarray]): Frames for visualization, note that
            len(frames) % len(annotations) should be 0.
        annotations (list[list[tuple]]): The predicted results.
        plate (str): The plate used for visualization. Default: plate_blue.
        max_num (int): Max number of labels to visualize for a person box.
            Default: 5.

    Returns:
        list[np.ndarray]: Visualized frames.
    """

    assert max_num + 1 <= len(plate)
    plate = [x[::-1] for x in plate]
    frames_ = cp.deepcopy(frames)
    nf, na = len(frames), len(annotations)
    assert nf % na == 0
    nfpa = len(frames) // len(annotations)
    anno = None
    h, w, _ = frames[0].shape
    scale_ratio = np.array([w, h, w, h])
    for i in range(na):
        anno = annotations[i]
        if anno is None:
            continue
        for j in range(nfpa):
            ind = i * nfpa + j
            frame = frames_[ind]
            for ann in anno:
                box = ann
                box = (box* scale_ratio).astype(np.int64)
                st, ed = tuple(box[:2]), tuple(box[2:])
                cv2.rectangle(frame, st, ed, plate[0], 2)
    return frames_


def parse_args():
    parser = argparse.ArgumentParser(description='MMAction2 demo')
    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')
    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')
    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')
    parser.add_argument('--video', help='video file/url')
    parser.add_argument(
        '--label-map', default='demo/label_map_ava.txt', help='label map file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    args = parser.parse_args()
    return args


def frame_extraction(video_path):
    """Extract frames given video_path.

    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into /tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    while flag:
        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)
        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()
    return frame_paths, frames


def detection_inference(args, frame_paths):
    """Detect human boxes given frame paths.

    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.

    Returns:
        list[np.ndarray]: The human detection results.
    """
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    results = []
    print('Performing Human Detection for each frame')
    for frame_path in tqdm(frame_paths):
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
    return results


def load_label_map(file_path):
    """Load Label Map.

    Args:
        file_path (str): The file path of label map.

    Returns:
        dict: The label map (int -> label name).
    """
    lines = open(file_path).readlines()
    lines = [x.strip().split(': ') for x in lines]
    return {int(x[0]): x[1] for x in lines}


def abbrev(name):
    """Get the abbreviation of label name:

    'take (an object) from (a person)' -> 'take ... from ...'
    """
    while name.find('(') != -1:
        st, ed = name.find('('), name.find(')')
        name = name[:st] + '...' + name[ed + 1:]
    return name


def pack_result(human_detection, result, img_h, img_w):
    """Short summary.

    Args:
        human_detection (np.ndarray): Human detection result.
        result (type): The predicted label of each human proposal.
        img_h (int): The image height.
        img_w (int): The image width.

    Returns:
        tuple: Tuple of human proposal, label name and label score.
    """
    human_detection[:, 0::2] /= img_w
    human_detection[:, 1::2] /= img_h
    results = []
    if result is None:
        return None
    for prop, res in zip(human_detection, result):
        res.sort(key=lambda x: -x[1])
        results.append(
            (prop.data.cpu().numpy(), [x[0] for x in res], [x[1]
                                                            for x in res]))
    return results


def main():
    args = parse_args()

    # frame_paths, original_frames = frame_extraction(args.video)

    

    video_pathes = os.listdir(args.video)
    # frame_paths = sorted([osp.join(osp.join(args.video, video_base_path), x) for video_base_path in video_pathes for x in os.listdir(osp.join(args.video, video_base_path)) ])

    # single folder
    # video_path=args.video
    # frame_paths = sorted([osp.join(video_path, x) for x in os.listdir(video_path)])

    for video_base_path in video_pathes:
        video_path=osp.join(args.video, video_base_path)
        frame_paths = sorted([osp.join(video_path, x) for x in os.listdir(video_path)])
    
        # original_frames = []
        # for x in os.listdir(video_path):
        #     frame=cv2.imread(osp.join(video_path, x))
        #     original_frames.append(frame)

        # num_frame = len(frame_paths)
        frame = cv2.imread(frame_paths[0])
        h, w, _ = frame.shape

        # Load label_map
        # label_map = load_label_map(args.label_map)

        # resize frames to shortside 256
        new_w, new_h = mmcv.rescale_size((w, h), (1800, np.Inf))
        # frames = [mmcv.imresize(img, (new_w, new_h)) for img in original_frames]
        w_ratio, h_ratio = new_w / w, new_h / h

        human_detections = detection_inference(args, frame_paths)
        for i in range(len(human_detections)):
            det = human_detections[i]
            det[:, 0:4:2] *= w_ratio
            det[:, 1:4:2] *= h_ratio
            human_detections[i] = torch.from_numpy(det[:, :4]).to(args.device)

        results_total = []
        for human_detection in human_detections:
            human_detection[:, 0::2] /= new_w
            human_detection[:, 1::2] /= new_h
            results = []
            for prop in human_detection:
                results.append((prop.data.cpu().numpy()))
            results_total.append(results)



        # xml
        target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
        os.makedirs(target_dir, exist_ok=True)
        for frame_path,anno in zip(frame_paths,results_total):
            output_name = os.path.join(target_dir,os.path.basename(frame_path))
            create_tree(frame_path)
            scale_ratio = np.array([w, h, w, h])
            if anno is None:
                continue
            for ann in anno:
                box = ann
                box = (box * scale_ratio).astype(np.int64)
                label="person"
                left, top, right, bottom = box.astype(float)
                create_object(annotation, label, left, top, right, bottom)


            tree = ET.ElementTree(annotation)
            root = tree.getroot()  # 得到根元素，Element类
            pretty_xml(root, '\t', '\n')  # 执行美化方法
            tree.write('%s.xml' % output_name.rstrip('.jpg'), encoding="utf-8")

    # vis_frames = visualize(frames, results_total)
    # vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames],
    #                             fps=6)
    # target_dir = osp.join('./tmp/test')
    # os.makedirs(target_dir, exist_ok=True)
    # frame_tmpl = osp.join(target_dir, 'img_%06d.jpg')
    # vid.write_images_sequence(frame_tmpl, fps=6)






if __name__ == '__main__':
    main()
