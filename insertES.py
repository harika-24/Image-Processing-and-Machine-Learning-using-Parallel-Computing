from datetime import datetime
from elasticsearch import Elasticsearch
import ast
import logging

logging.basicConfig(level=logging.DEBUG)

es = Elasticsearch()



with open('/home/sethiamayank14/PycharmProjects/project2/src/allData.json','r') as datafile:
    data = datafile.read().split('\n')
x = 0
for item in data:
    value = {}
    item = ast.literal_eval(item)
    value.update({"name":item["face-bv"]["name"],"embedding_vector":item["face-bv"]["embedding_vector"]})

    try:
        es.index(index='face-bv', doc_type='test', id=x, body=value)
        x = x + 1
    except:
        pass


print("successful")