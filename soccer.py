#!/usr/bin/python3
from SoccerNet.Downloader import SoccerNetDownloader
from io import TextIOWrapper
import numpy as np
import collections
import zipfile
import csv
import cv2

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="data/SoccerNet")
mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train","test","challenge"])

def get_np_array_from_tar_object(tar_extractfl):
     '''converts a buffer from a tar file in np.array'''
     return np.asarray(bytearray(tar_extractfl.read()), dtype=np.uint8) 

def collect(fn):
    archive = zipfile.ZipFile(fn, 'r')
    
    annotated_images = []

    annotation_files = []
    for f in archive.namelist():
        if "det/det.txt" in f:
            annotation_files.append(f)

    image_annotations = collections.defaultdict(list)

    for af in annotation_files:
        st, sample,_,__ = af.split("/")
        imgpath = "%s/%s/img1/%06d.jpg"
        for row in csv.reader(TextIOWrapper(archive.open(af), 'utf-8') ):
            frame, _, x, y, w, h = [int(x) for x in row[:6]] 
            ifn = imgpath%(st,sample, frame)
            image_annotations[ifn].append((x,y,w,h))
            
    for k in image_annotations.keys():
        img = cv2.imdecode(get_np_array_from_tar_object(archive.open(k)), 1 )
        for box in image_annotations[k]:
            x,y,w,h = box
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0), 2)

        img = cv2.resize(img, (0,0), fx=0.75, fy=0.75) 
        cv2.imshow('image',img)
        cv2.waitKey(100)
            
train = collect("path/to/SoccerNet/tracking/train.zip")
cv2.destroyAllWindows()    
        