
import numpy as np
import collections
from scipy.ndimage.interpolation import zoom
import os
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
import cv2
import glob
from sklearn.decomposition import PCA, NMF, FastICA

import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')

# Load model
# from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import Model

from Autoencoders import DeepAutoencoder, Autoencoder, pdAutoEncoder

def featureextraction(rotate_img_list, model):
    batch_img_list = list()
    for image in rotate_img_list:
        # rot_images = zoom(image, zoom=[224/image.shape[0], 224/image.shape[1]], order=3)
        rot_images = zoom(image, zoom=[331/image.shape[0], 331/image.shape[1]], order=3)
        rot_images = np.array(rot_images,dtype=np.float)    
        x___ = np.stack((rot_images,)*3, axis=-1)
        x__ = np.expand_dims(x___, axis=0)
        batch_img_list.append(x__)
    
    x_ = np.concatenate(batch_img_list, axis=0)
    x = preprocess_input(x_)
    base_model_pool_features = model.predict(x)
        
    concat_features_list = list()
    for rot in range(base_model_pool_features.shape[0]):
        feature_map = base_model_pool_features[rot]
        feature_map = feature_map.transpose((2,1,0))
        features = np.std(feature_map,axis=(1,2))
        concat_features_list.append(features.reshape(1, -1))
    concat_features = np.concatenate(concat_features_list, axis=0)
    final_feature = np.hstack(concat_features)
    final_feature = final_feature.transpose()
    deeplearningfeatures = collections.OrderedDict()
    for ind_,f_ in enumerate(final_feature[:]):
    	deeplearningfeatures[str(ind_)] = f_
    return deeplearningfeatures



def main(model, DataDir, csv_filename):
    
    # Load batch file    
    fileformat = f"*_CROPPED_PET_MIP-5.png"
    ImgDir = os.path.join(DataDir, fileformat)
    dirlist = glob.glob(ImgDir)
    
    # Feature Extraction
    featureDict = {}
    ind = 0
    for dirname in dirlist:
        
        patient_ID = os.path.basename(dirname)[0:-22]
        img_rotations_list = list()
        for rotation in range(0, 360, 5):
            
            imgpath = os.path.join(DataDir, f"{patient_ID}_CROPPED_PET_MIP-{rotation}.png")
            img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
            img_rotations_list.append(img)
            
        result = featureextraction(img_rotations_list, model) 

        key = list(result.keys())
        key = key[0:]
            
        feature = []
        for jind in range(len(key)):
            feature.append(result[key[jind]])
            
        featureDict[patient_ID] = feature
        dictkey = key
        print(patient_ID)

        ind += 1
            
    dataframe = pd.DataFrame.from_dict(featureDict, orient='index', columns=dictkey)
    AE = pdAutoEncoder(dataframe)
    AE.fit(epochs=500, batch=100)
    fused_dataframe = AE.encode()
    fused_dataframe.to_csv(csv_filename)
    return


if __name__ == "__main__":
    
    model = DenseNet201(weights='imagenet', include_top=False)
    
    NETWORK = "DenseNet201"
    
    PET_CT = 'PET'
    DataDir = '/datassd/WHOLEIMAGE_MAMIP/DATA/MIP/PET'
    FeaturesDir = '/datassd/WHOLEIMAGE_MAMIP/EXTRACTED_FEATURES/ExtractedFeatures_STD-AE'
    if not os.path.exists(FeaturesDir):
        os.makedirs(FeaturesDir)
    csv_filename = os.path.join(FeaturesDir, f"Features_{NETWORK}_MA-MIP_WHOLE-IMAGE.csv")
    # ifile = f"{NETWORK}.jpg"
    # plot_model(model, to_file = ifile, show_shapes = True, show_layer_names = True)
    main(model, DataDir, csv_filename)

