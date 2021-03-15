视频分类数据生成步骤
#抽取视频帧
python build_rawframes.py ../../data/ucf_crime_video ../../data/ucf_crime --task rgb --level 2 --ext mp4 --use-opencv
#生成文件标签列表
python vid2img_kinetics.py ../../data/ucf_crime_video/ ../../data/ucf_crime
#生成训练集和验证集
python gen_label_kinetics.py
ps:参数写在里面，需要手动修改路径

时空动作检测数据生成步骤（仿照ava数据集的格式）
#抽取视频帧
bash extract_rgb_frames_ffmpeg.sh
ps:需要手动修改里面的路径，extract_rgb_frames_opencv脚本碰到视频帧错误会停止继续抽取图片帧操作。

#根据已有标注生成类似ava格式的标注
python gen_ava_dataset.py
ps:需要手动修改里面的路径