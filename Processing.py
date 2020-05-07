#!/usr/bin/python


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image 
import openslide
import sys
import cv2
import random
import shutil
from time import sleep
import time
import json
import os
import glob
from PIL import ImageEnhance
import torch
import torch.nn.functional as F
sys.path.append("Functions")
from deephistopath.wsi import slide
from sklearn.feature_extraction import image
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Constants
OVERLAP = 0.5
NB_COMPONENTS_PCA = 128


class PreProcessing():
    """Class consisting of all preprocessing methods of the png files from the WSI slides.
    Attributes:
        list_img_path: list of the png files paths.
        model: optional, the preprocessing model for extracting fetures from the image.
        overlap: Number between 0 and 1 representing the overlap between patches from a same image.
        pca_comp: The number of components for the PCA done over all patches from the same image.
        keys: the list of keys. A key is an easy identifiable name representation of the slide number and patient number.
    """
    
    def __init__(self, list_img_path, model=None):
        self.list_img_path = list_img_path
        self.model = model
        self.overlap = OVERLAP
        self.pca_comp = NB_COMPONENTS_PCA
        self._keys = self.keys()
    
    def keys(self):
        """Convert the list of paths to a list of keys.
        Return : A list of easy identifiable name representation of the slide number and patient number. 
        """
        return pd.Series(self.list_img_path).map(lambda x : (x.split(".")[0]+'.'+x.split(".")[1]).split("_")[1:])
    
    @staticmethod
    def labeling(key):
        """ Using the key representation to create labeling of slides.
        Parameters : A key.
        Return : -1 : Not HE stained slides.
        0 : Slides without tumors.
        1 : Slides with tumors.
        2 : HE staining slides we don't know if it contain tumor or not.
        """
        if "HE" not in key:
            return -1
        elif 'a' in list(key[0]):
            return 0
        elif ('b' in list(key[0])) or ('c' in list(key[0])) or ('d' in list(key[0])):
            return 1
        else : 
            return 2       
    
        
    @staticmethod
    def binary_mask(img_path):
        """ From the path of an image convert it to a binary mask to differentiate the background.
        Parameter : The path of a slide.
        Return : The binary mask.
        """
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        sure_bg = cv2.bitwise_not(cv2.dilate(opening,kernel,iterations=3))
        return sure_bg
    
    @staticmethod
    def roi(wat_img, img_path):
        """ Center the image into the region of interest (without the background).
        Parameters : wat_img : a binary mask, 
        img_path : the path of the slide associated to it.
        Return : The region of interest.
        """
        contours,hierarchy = cv2.findContours(wat_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(wat_img, contours, 0, (0, 255, 0), 2)
        max_area = 0
        idx = 0
        for i in range(len(contours)):
            if(cv2.contourArea(contours[i])>max_area):
                max_area = cv2.contourArea(contours[i])
                idx = i
        cnt = contours[idx]
        x,y,w,h = cv2.boundingRect(cnt)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        roi = img[y:y+h, x:x+w]
        return Image.fromarray(roi)
    
    @staticmethod
    def slicing_images(img, shape, overlap):
        """ Slice an image to multiple squares of defined size with an overlap.
        Parameters : img : a slide image,
        shape : the size of the squares (shape*shape),
        overlap : the amount of overlap between the squares ranging between 0 and 1.
        Return : sliced : the list of patches.
        vertices : the left-up corner of the square in coordinates.
        """
        y = 0
        sliced = []
        vertices = []
        while y+shape < img.shape[1]:
            x = 0
            while x+shape < img.shape[0]:
                sliced.append(img[x:x+shape, y:y+shape])
                vertices.append([x, y])
                x += int(shape*(1-overlap))
            sliced.append(img[img.shape[0]-shape:img.shape[0], y:y+shape])
            vertices.append([img.shape[0]-shape,y])
            y += int(shape*(1-overlap))
        x = 0
        while x+shape < img.shape[0]:
            sliced.append(img[x:x+shape, img.shape[1]-shape:img.shape[1]])
            vertices.append([x,img.shape[1]-shape])
            x += int(shape*(1-overlap))
        sliced.append(img[img.shape[0]-shape:img.shape[0], img.shape[1]-shape:img.shape[1]])
        vertices.append([img.shape[0]-shape, img.shape[1]-shape])
        return sliced, vertices
    
    def imgs_to_patches(self, path):
        """ From the slide image create 224*224 squares patches centered around the region of interest.
        Parameter: the slide path
        Return : A list of all patches sliced from the image.
        """
        wat_img = self.binary_mask(path)
        roi = self.roi(wat_img, path)
        patch, vertex = self.slicing_images(np.array(roi), 224, self.overlap)
        return patch
    
    @staticmethod
    def black_proportion(img):
        """ Calculate the proportion of black pixels from an image.
        Parameters: the image.
        Return: the black propotion of the image.
        """
        pixels = Image.fromarray(img).getdata()
        n = len(pixels)
        nblack = 0
        for pixel in pixels:
            if pixel == (0,0,0):
                nblack += 1
        return (nblack/n)
    
    def extract_features(self, patches, number):
        """ From a patch return the features extracted from the model self.model.
        Parameters : patches : The list of patches.
        number : The patch number in the list of patches.
        Return : The features extracted from the model.
        """
        final_patch = patches[number].reshape((1, patches[number].shape[0], patches[number].shape[1], patches[number].shape[2]))
        final_patch = preprocess_input(final_patch)
        features = self.model.predict(final_patch)
        return features
    
    def save_to_features(self, patches, keys):
        """ From the patches extract the features.
        Parameters : patches : The list of patches.
        keys : The keys representation.
        Return : Create a document with the reduced features from all patches using keys as name of the file.
        """
        features = pd.DataFrame(range(len(patches)), columns =['features']).applymap(lambda x : self.extract_features(patches, x))
        features = features.explode("features")
        feat = pd.DataFrame(features["features"].values.tolist())
        feat.to_csv('root/features_dir/'+'_'.join(keys)+'.csv')
        shutil.move("data/thumbnail_"+'_'.join(keys)+".mrxs.png", "data2/thumbnail_"+'_'.join(keys)+".mrxs.png")
            
    def imgs_to_features(self):
        """ From the path of the png files create a document for each with the features extracted.
        Return : The documents containing the features.
        """
        
        len_time = len(self.list_img_path)*0.05
        for i, path in enumerate(self.list_img_path):
            if (self.labeling(self._keys[i]) == 0) or (self.labeling(self._keys[i]) == 1):
                patches = self.imgs_to_patches(path)
                int_patches = list(filter(lambda x : self.black_proportion(x)<0.3, patches))
                self.save_to_features(int_patches, self._keys[i])
                sys.stdout.write('\r')
                sys.stdout.write("[%-20s] %d%%" % ('='*int(i/len_time), int(5*i/len_time)))
                sys.stdout.flush()
                sleep(0.25)
            
            
            
    def pca_reduc(self):
        """ Performs principal components analysis on features from slides. 
        Return : Save the csv files after PCA reduction to the raw_dir folder.
        """
        
        pca = PCA(n_components = 128)
        feat_csv = glob.glob("root/features_dir/*.csv")
        feat = []
        for files in feat_csv:
            feat.append(pd.read_csv(files))
        df_feat = pd.concat(feat, axis=0)
        pca.fit(df_feat.values[:,1:])
                              
        for i, df in enumerate(feat):
            df_final_feat = pd.DataFrame(pca.transform(df.values[:,1:]), columns = ['PCA%i' %i for i in range(128)], index = df.index)
            df_final_feat.to_csv('root/raw_dir/'+feat_csv[i].split("/")[-1])
            
            
            

    @staticmethod
    def dataset_splitting(final_keys, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        """ Create a root directory with the features that are split into train, validation and test folders
        Parameters : final_keys : the keys format of the saved dataset
        train_ratio, val_ratio, test_ratio : The ratio of training, validation and testing data respectively 
        (default values are 60%, 20% and 20% respectively).
        Return : The tree folders training, validation and testing
        """
        
        x_train, x_test = train_test_split(final_keys["Name"], test_size = 1-train_ratio)
        x_val, x_test = train_test_split(x_test, test_size = test_ratio/(val_ratio+test_ratio))
        
        for train in x_train:
            shutil.move('root/raw_dir/' + "/" + "_".join(train) + ".csv", 'root/raw_dir/' + "/train/" + "_".join(train) + ".csv")
        for val in x_val:
            shutil.move('root/raw_dir/' + "/" + "_".join(val) + ".csv", 'root/raw_dir/' + "/validation/" + "_".join(val) + ".csv")
        for test in x_test:
            shutil.move('root/raw_dir/' + "/" + "_".join(test) + ".csv", 'root/raw_dir/' + "/test/" + "_".join(test) + ".csv")
 
           
            
class PostProcessing():
    """ Class allowing to visualize the result of the graph convolutional neural network on a single slide.
    Attributes:
        csv_file_path: The path of the csv file where the features are stocked.
        model: The trained pytorch model giving the best result on binary classification with a sigmoid layer for each patch.
        roi: the region of interest of the slide.
        vertices: the list of left-upper coordinates from patches of the same slide.
    """

    
    def __init__(self, csv_file_path, model):
        self.csv_file_path = csv_file_path
        self.model = model
        self.module = fe.features_engineering(glob.glob("data2/*.png"))
        self.roi = self.to_roi()
        self.vertices = self.get_vertices()
         
    def to_roi(self):
        """ Create the region of interest of the slide.
        Return : the region of interest.
        """
        img_file_path = "data2/thumbnail_" + ".".join(self.csv_file_path.split('/')[-1].split('.')[0:2]) + ".mrxs.png"
        wat_img = self.module.binary_mask(img_file_path)
        roi = self.module.roi(wat_img, img_file_path)
        patches, vertices = self.module.slicing_images(np.array(roi), 224, self.module.overlap)
        for i,patch in enumerate(patches):
            if self.module.black_proportion(patch) > 0.3:
                patch = Image.new('RGB', (224, 224))
                box = (vertices[i][1], vertices[i][0], vertices[i][1]+224, vertices[i][0]+224)
                roi.paste(patch, box)
        return roi
    
    def get_vertices(self):
        """ Get the list of of left-upper coordinates from patches of the same slide.
        Return: The list of of left-upper coordinates from patches of the same slide.
        """
        img_file_path = "data2/thumbnail_" + ".".join(self.csv_file_path.split('/')[-1].split('.')[0:2]) + ".mrxs.png"
        wat_img = self.module.binary_mask(img_file_path)
        roi = self.module.roi(wat_img, img_file_path)
        patches, vertices = self.module.slicing_images(np.array(roi), 224, self.module.overlap)
        idx = np.array([self.module.black_proportion(x)<0.3 for x in patches])
        vertices = np.array(vertices)
        vertices = vertices[idx]
        return vertices

    def get_coordinates(self, number):
        """ Calculate the corresponding left-up corner coordinate of the patch corresponding to the number from an image. Used to see which patches are connected in the graph or what patchs have high attention in the model.
        Arguments : number : the index in the list of patches.
        Return : the coordinates and the corresponding patches.
        """
        return self.vertices[number], np.array(self.roi)[self.vertices[number][0]:self.vertices[number][0]+224,self.vertices[number][1]:self.vertices[number][1]+224,:]
    
    def attention_images(self, number, alpha):
        """ Get the region of interest with brightness that depend on where the attention of the model is focus.
        High brghtness means parts of the image having received a large weight from the attention network.
        Arguments : number : the index in the list of patches.
        alpha : the weights given on particular patches.
        Return : The orignal image with different level of brightness.
        """
        corner, patch = self.get_coordinates(number)
        box = (corner[1], corner[0], corner[1]+224, corner[0]+224)
        bright = ImageEnhance.Color(Image.fromarray(patch))
        crop = bright.enhance(alpha)  
        self.roi.paste(crop, box)
    
    def tumors_scan(self):
        """ Get the patches where the probability that the patch being tumoral is higher than 90% in colors and the other patches in black and whithe. Allows to visualize where the tumor is located on the slide.
        Return: The slides where the computed tumors are in colors and the healthy tissue in black and white.
        """
        data = torch.load("root/processed_dir/" + ".".join("/".join(self.csv_file_path.split('/')[-2:]).split('.')[0:2]) + ".pt")
        x, edge_index = data.x, data.edge_index.type(torch.LongTensor)
        
        self.model.eval()
        x = self.model(x)
        x = x.detach().cpu().numpy()
        
        for number, pred  in enumerate(x):
            if pred > 0.9:
                self.attention_images(number, 1)
            else:
                self.attention_images(number, 0)
         
            

   