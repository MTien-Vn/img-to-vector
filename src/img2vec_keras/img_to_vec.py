import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import clip
import traceback
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

_HEIGHT = 512
_IMAGE_NET_TARGET_SIZE = (224, 224)


class Img2Vec(object):
    def __init__(self):
        self.default_vector = self.get_default_vector()
        # self.model = TFAutoModel.from_pretrained("openai/clip-vit-base-patch32")
        # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model, self.preprocess = clip.load('ViT-B/32')


    def get_vec(self, image_path):
        """ Gets a vector embedding from an image.
        :param image_path: path to image on filesystem
        :returns: numpy ndarray
        """
        try:
            fileName = image_path.split('/')[5]
            image_url = tf.keras.utils.get_file('img' + fileName, origin=image_path )
            img = image.load_img(image_url, target_size=_IMAGE_NET_TARGET_SIZE )
            # os.remove(image_url)
            image_array = image.img_to_array(img)
            
            image_array = self.preprocess(image_array)
            image_array = tf.expand_dims(image_array, axis=0)
            embedded = self.model.encode(image_array)
            
            return embedded.numpy().toList()
        except Exception as e:
            print(f"{image_path} Error get_vec: {e}")
            traceback.print_exc()
            return self.default_vector
        # return x

    def get_default_vector(self):
        vector = []
        for i in range(0, _HEIGHT):
            vector.append(0)
        return vector
        
if __name__ == "main":
     pass    