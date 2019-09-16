# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 00:57:56 2019

@author: Akankshya
"""

#####get features
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import scipy.misc
from tqdm import tqdm
from moviepy.editor import VideoFileClip

def crop_center(im):
    """
    Crops the center out of an image.
   
    Args:
     im (numpy.ndarray): Input image to crop.
    Returns:
     numpy.ndarray, the cropped image.
    """
   
    h, w = im.shape[0], im.shape[1]
   
    if h < w:
     return im[0:h,int((w-h)/2):int((w-h)/2)+h,:]
    else:
     return im[int((h-w)/2):int((h-w)/2)+w,0:w,:]


def extract_features(input_dir, output_dir, model_type='vgg16', batch_size=32):
    """
    Extracts features from a CNN trained on ImageNet classification from all
    videos in a directory.
   
    Args:
     input_dir (str): Input directory of videos to extract from.
     output_dir (str): Directory where features should be stored.
     model_type (str): Model type to use.
     batch_size (int): Batch size to use when processing.
    """
    print ("Feature is extracted" )
    input_dir = os.path.expanduser(input_dir)
    output_dir = os.path.expanduser(output_dir)
   
    if not os.path.isdir(input_dir):
        sys.stderr.write("Input directory '%s' does not exist!\n" % input_dir)
        sys.exit(1)
   

 # Load desired ImageNet model
 
 # Note: import Keras only when needed so we don't waste time revving up
 #       Theano/TensorFlow needlessly in case of an error

    model = None
    shape = (224, 224)
   
    from keras.applications import VGG16
    model = VGG16(include_top=True, weights='D:/final_yr_project/2videocaption/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    
    # Get outputs of model from layer just before softmax predictions
   
    from keras.models import Model
    model = Model(model.inputs, output=model.layers[-2].output)
   
   
    # Create output directories
   
    visual_dir = os.path.join(output_dir, 'visual') # RGB features
    #motion_dir = os.path.join(output_dir, 'motion') # Spatiotemporal features
    #opflow_dir = os.path.join(output_dir, 'opflow') # Optical flow features
   
    for directory in [visual_dir]:#, motion_dir, opflow_dir]:
        if not os.path.exists(directory):
           os.makedirs(directory)


 # Find all videos that need to have features extracted

    def is_video(x):
       return x.endswith('.mp4') or x.endswith('.avi') or x.endswith('.mov')
   
    vis_existing = [x.split('.')[0] for x in os.listdir(visual_dir)]
    #mot_existing = [os.path.splitext(x)[0] for x in os.listdir(motion_dir)]
    #flo_existing = [os.path.splitext(x)[0] for x in os.listdir(opflow_dir)]
   
    video_filenames = [x for x in sorted(os.listdir(input_dir)) if is_video(x) and os.path.splitext(x)[0] not in vis_existing]
   
   
    # Go through each video and extract features
   
    from keras.applications.imagenet_utils import preprocess_input
    print ("Feature is extracted" )
    for video_filename in tqdm(video_filenames):
   
     # Open video clip for reading
        try:
           clip = VideoFileClip( os.path.join(input_dir, video_filename) )
        except Exception as e:
            sys.stderr.write("Unable to read '%s'. Skipping...\n" % video_filename)
            sys.stderr.write("Exception: {}\n".format(e))
            continue
   
     # Sample frames at 1fps
        fps = int( np.round(clip.fps) )
        frames = [scipy.misc.imresize(crop_center(x.astype(np.float32)), shape) for idx, x in enumerate(clip.iter_frames()) if idx % fps == fps//2]
      
        print ("Feature is extracted" )
        n_frames = len(frames)
      
        frames_arr = np.empty((n_frames,)+shape+(3,), dtype=np.float32)
        for idx, frame in enumerate(frames):
           frames_arr[idx,:,:,:] = frame
      
        frames_arr = preprocess_input(frames_arr)
      
        features = model.predict(frames_arr, batch_size=batch_size)
      
        name, _ = os.path.splitext(video_filename)
        feat_filepath = os.path.join(visual_dir, name+'.npy')
        print ("Feature is extracted" )
        with open(feat_filepath, 'wb') as f:
           np.save(f, features)


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Extract ImageNet features from videos.")
   
    parser.add_argument('-i', '--input',default='D:/final_yr_project/2videocaption/video', type=str,help="Directory of videos to process.")
    parser.add_argument('-o', '--output',default='D:/final_yr_project/2videocaption/video' ,type=str, help="Directory where extracted features should be stored.")
   
    parser.add_argument('-m', '--model', default='vgg16', type=str,help="ImageNet model to use.")
    parser.add_argument('-b', '--batch_size', default=10, type=int,help="Number of frames to be processed each batch.")
   
    args = parser.parse_args()
   
    extract_features(input_dir=args.input, output_dir=args.output,model_type=args.model, batch_size=args.batch_size)
