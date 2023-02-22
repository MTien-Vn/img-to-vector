import json
from flask import Flask, request
from flask_caching import Cache
from img2vec_keras import Img2Vec

config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300
}

app = Flask(__name__)
# tell Flask to use the above defined config
app.config.from_mapping(config)
cache = Cache(app)

@cache.cached(timeout=50, key_prefix='init_img2vector')
def init_img2vector():
  img2vec = Img2Vec()
  return img2vec


@app.route('/img2vector', methods=["POST"])
def img2vector():
  url = request.json.get('url')
  img2vec = init_img2vector()
  dense_vector = img2vec.get_vec(url)
  # json_dump = json.dumps(dense_vector)
  res = {
    "dense-vector": dense_vector
  }
  return res

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5001)