#coding:utf-8

import shutil
import os
 
new_dir_file='newFrames/'
ann_dir='/home/lishuang/Disk/gitlab/traincode/mmaction2/data/jhmdb/Frames/'
i=0
for root, _, files in os.walk(ann_dir):
    for filename in files:
        if filename.endswith('.png'):
            imagename=filename.replace('.png','.jpg')
            new_dir=root.replace("Frames/",new_dir_file)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            newdir=os.path.join(new_dir,'img_'+imagename)
            #print(newdir)
            if i%100==0:
                 print("Processed %d images" % (i))
            shutil.copyfile(os.path.join(root,filename),newdir)
            i=i+1
            #os.rename(root+filename,newdir)

