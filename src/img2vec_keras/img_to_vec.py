import os
import tensorflow as tf
from tensorflow.keras.applications import resnet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

_HEIGHT = 512
_WIDTH = int(2048 / _HEIGHT)
_IMAGE_NET_TARGET_SIZE = (224, 224)
_RESHAPE = (_HEIGHT,_WIDTH)


class Img2Vec(object):
    def __init__(self):
        self.default_vector = self.get_default_vector()
        model = resnet50.ResNet50(weights='imagenet')
        # model.save("weights.h5")
        # model = load_model("weights.h5")
        layer_name = 'avg_pool'
        self.intermediate_layer_model = Model(inputs=model.input, 
                                              outputs=model.get_layer(layer_name).output)


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

            # img = image.load_img(image_path, target_size=_IMAGE_NET_TARGET_SIZE)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = resnet50.preprocess_input(x)
            intermediate_output = self.intermediate_layer_model.predict(x)
            dense_vector = intermediate_output[0]
            dense_vector = dense_vector.reshape(_RESHAPE)
            dense_vector = tf.reduce_mean(dense_vector, 1)
            return dense_vector.numpy().tolist()
        except:
            print("Error get_vec: ", image_path)
            return self.default_vector
        # return x

    def get_default_vector(self):
        vector = []
        for i in range(0, _HEIGHT):
            vector.append(0)
        return vector
        
if __name__ == "main":
     pass    