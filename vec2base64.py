import csv
import base64
import numpy as np
from numpy import array

dbig = np.dtype('>f8')

def decode_float_list(base64_string):
    bytes = base64.b64decode(base64_string)
    return np.frombuffer(bytes, dtype=dbig).tolist()

def encode_array(arr):
    base64_str = base64.b64encode(np.array(arr).astype(dbig)).decode("utf-8")
    return base64_str

with open("test.csv") as f:
        reader = csv.reader(f)
        data = [r for r in reader]

writeFile = open('allData.json','w')
finaljson = dict()
valueDict = dict()
index = "face-bv"

for x in range(1,len(data)-1):
        name = (data[x][0])
        a = array((data[x][1:]))
        encoded_array = encode_array(a)
        print("name" + name)
        print("Encoded Array" + encoded_array)

        valueDict.update({"index":index})
        valueDict.update({"name":name})
        valueDict.update({"embedding_vector":encoded_array})

        finaljson.update({index:valueDict})
        writeFile.write(str(finaljson)+'\n')
