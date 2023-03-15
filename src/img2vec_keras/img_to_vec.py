import os
import requests
from io import BytesIO
from sentence_transformers import SentenceTransformer
from PIL import Image
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

_HEIGHT = 512
class Img2Vec(object):
    def __init__(self):
        self.default_vector = self.get_default_vector()
        self.img_model = SentenceTransformer('clip-ViT-B-32')


    def get_vec(self, image_path):
        """ Gets a vector embedding from an image.
        :param image_path: path to image on filesystem
        :returns: numpy ndarray
        """
        try:
            response = requests.get(image_path)
            if response.status_code == 200 and response.content is not None:
                image = Image.open(BytesIO(response.content))
                embedding = self.img_model.encode(image)
                return embedding.tolist()
            else:
                return self.default_vector
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