from __future__ import print_function
import click
import os
import re
import face_recognition.api as face_recognition
import multiprocessing
import itertools
import sys
import PIL.Image
import numpy as np
import os
import sys
import dlib
import glob
import csv
import pickle as pp
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing
# from sklearn.model_selection import train_test_split
import webbrowser
from timeit import Timer
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
from time import time
from flask import Flask, render_template, request
from PIL import Image
from elasticsearch import Elasticsearch
from tensorflow.python.keras._impl.keras.preprocessing.image import img_to_array
from flask import Flask, render_template, request, url_for

app = Flask(__name__, template_folder='templates')

App_root=os.path.dirname("maintype")
@app.route("/knn")

def classify(try_vector):                                                #CLASIFIER OPTION -A using KNN
    print("in classifier======================================================")
    p_1=pp.load(open('model.p','rb'))
    p_2=pp.load(open('model_1.p','rb'))
    pred = p_1.predict([try_vector])
    v = p_2.inverse_transform(pred)
    print(p_2.inverse_transform(pred))
    return v

# @app.route("/cnn")                                          ##CNN FACE IDENTIFICATION SYSTEM
# def conv():
#     des1 = pp.load(open('model_3.p', 'rb'))./cnn_face_detector.py mmod_human_face_detector.dat ../examples/faces/*.jpg
#     im = Image.open(des1);
#     classes=pp.load(open('model_5.p','rb'))
#     imrs = im.resize((50,50))
#     imrs = img_to_array(imrs) / 255;
#     imrs = imrs.transpose(2, 0, 1);
#     imrs = imrs.reshape(3, 50, 50);
#     x = []
#     x.append(imrs)
#     x = np.array(x);
#     model=load_model('basic_model.h5')
#     predictions = model.predict(x)
#     print(predictions)
#     l=list(predictions[0])                                                      ##list of the prediction
#     a=max(l)
#     n=l.index(a)                                                              ##finding the index of the list
#     print(a)
#     return classes[n]

#@app.route("/ess")
# def esearch(try_vector):
#     es = Elasticsearch()
#     # del1 = pp.load(open('model_3.p', 'rb'))
#     # face_vector =list(vector(del1))
#     face_vector = try_vector
#     res = es.search(index="face-bv", body={
#             "query": {
#                 "function_score": {
#                     "boost_mode": "replace",
#                     "script_score": {
#                         "script": {
#                             "inline": "binary_vector_score",
#                             "lang": "knn",
#                             "params": {
#                                 "cosine": False,
#                                 "field": "embedding_vector",
#                                 "vector": face_vector
#                             }
#                         }
#                     }
#                 }
#             },
#             "size": 5
#         })
#     print(res)
#     return (res['hits']['hits'][1]['_source']['name'])
#     # return "Elastic-search"




def vector(destination,option):                                                ###CONVERTING IMAGE INTO 128 vectors --DLIB
    predictor_path = "shape_predictor_5_face_landmarks.dat"
    face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
    faces_folder_path ="/home/sethiamayank14/PycharmProjects/project2/src/"+destination
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)
    img = dlib.load_rgb_image(faces_folder_path)
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = sp(img, d)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        try_vector=face_descriptor
        #print("======================================",try_vector)
        if option == "KNN":
            starttime = time()
            d = classify(try_vector)  #knn
            timetaken = time()-starttime
            print(timetaken)
        elif option == "ES":
            print("ES Selected")
            return ("ES Selected")
             # d = esearch(try_vector)
        else:
            print("CNN Chosen")

    return d







@app.route("/")                           # this runs first
def index():
    print("index working==================================")
    return render_template("upload1.html")

@app.route("/upload", methods = ['POST'])
def upload():
    # print("heyy========================")
    target = os.path.join(App_root, "images/")
    # print("hello")
    if not os.path.isdir(target):
        print("In here")
        os.mkdir(target)
    print("-----------------------",request.files.getlist("file"))
    for file in request.files.getlist("file"):
        filename = file.filename
        destination ="".join([target, filename])
        print(destination)
        file.save(destination)
        option = request.form['classifier']
        print(option)
        if( option == "KNN" or option == "ES"):
            name1 = vector(destination,option)
            name1 = str(name1[0])
            print(name1, type(name1))

            f = open('helloworld.html', 'w')
            # # name = "Akash Bhaiya"
            name = name1 + '.jpg'
            print(name)
            name2 = "/home/sethiamayank14/PycharmProjects/project2/src/images/" + name
            print(name2)
            message = """<html>
            <head></head>
            <body>
            <p>Your input image: </p>
            <br>
            <img src = "/home/sethiamayank14/PycharmProjects/project2/src/""" + destination + """"/>
            <br>
            <p>Standard Image:</p>
            <br>
            <img src = "/home/sethiamayank14/PycharmProjects/project2/src/images/""" + name + """"/>
            <p>  """ + name1 + """</p>
            </body>
            </html>"""
            print(message)
            f.write(message)
            f.close()

            # Change path to reflect file location
            filename = 'helloworld.html'
            webbrowser.open_new_tab(filename)
            return name

            # return name


        else:
            # print("CNN Chosen")
            # return("Hello world")
            name = vector(destination, option)
            name = str(name[0])
            print(name, type(name))
            f = open('complete.html', 'w')
            message = name

            f.write(message)
            f.close()
            # with open("complete.html", "w") as file1:
            # file1.write(html)
            # return render_template("complete.html")
            return("This is CNN Button")






if __name__== "__main__":
    app.run(debug=True,port=5001,host='127.0.0.1')

