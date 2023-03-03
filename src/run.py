import json
from flask import Flask, request
import concurrent.futures
import time
app = Flask(__name__)

from img2vec_keras import Img2Vec

@app.route('/img2vector', methods=["POST"])
def img2vector():
  img2vec = Img2Vec()
  urls = request.json.get('urls')
  urls_dic = json.loads(urls)
  # for id in urls_dic.keys():
  #   url = urls_dic[id]
  #   if(url):
  #     dense_vector = img2vec.get_vec(url)
  #     urls_dic[id] = dense_vector
  #   else:
  #     urls_dic[id] = img2vec.get_default_vector()


  # first way, using multithread
  result = {}
  start_time = time.perf_counter()
  with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
    for key, value in zip(urls_dic.keys(), executor.map(img2vec.get_vec, urls_dic.values())):
        result.update({
          key: value
        })
    # executor.map(img2vec.get_vec, urls_dic.values())
  finish_time = time.perf_counter()
  print("Program finished in {} seconds - using multithread".format(finish_time-start_time))
  print("---")

  # json_dump = json.dumps(urls_dic)
  res = {
    "dense_vectors": result
  }
  return res

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5001)