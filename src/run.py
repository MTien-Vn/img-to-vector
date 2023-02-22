import json
from flask import Flask, request
app = Flask(__name__)

from img2vec_keras import Img2Vec

@app.route('/img2vector', methods=["POST"])
def img2vector():
  url = request.json.get('url')
  img2vec = Img2Vec()
  dense_vector = img2vec.get_vec(url)
  # json_dump = json.dumps(dense_vector)
  res = {
    "dense-vector": dense_vector
  }
  return res

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5001)