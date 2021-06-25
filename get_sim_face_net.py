from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import os
from PIL import Image
import torch.nn as nn

def preprocess(path=None,im=None):
    if path is not None:
        tmp = np.array(Image.open(path))
    else:
        tmp = im
    
    if len(tmp.shape) > 2:
        tmp = np.transpose(tmp,(2,0,1))
    
    tmp = torch.FloatTensor(tmp)
    tmp -= tmp.min()
    tmp /= tmp.max()
    
    tmp = tmp * 2 - 1
    tmp = nn.Upsample(size = (160,160))(tmp.expand(3,tmp.shape[-1],tmp.shape[-1])[None,...])
    return tmp

def get_sim(device, path1=None, path2=None, im1=None, im2=None):
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6,0.7,0.7], factor=0.709,
        device=device
    )
    
    with torch.no_grad():
        if path1 is not None:
            emb1 = resnet(preprocess(path=path1).to(device))
        else:
            emb1 = resnet(preprocess(im=im1).to(device))
            
        if path2 is not None:
            emb2 = resnet(preprocess(path=path2).to(device))
        else:
            emb2 = resnet(preprocess(im=im2).to(device))
            
        return torch.cosine_similarity(emb1,emb2).item()
        