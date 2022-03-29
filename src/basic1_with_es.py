import os
import sys
import dlib
import glob
import csv
import pickle as pp
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from flask import Flask,render_template,request
import requests
from elasticsearch import Elasticsearch
import json

@app.route("/es")
def elasticSearch():
    es = Elasticsearch()
    face_vector = vector(destination)

    res = es.search(index="face-bv", body={
      "query": {
        "function_score": {
          "boost_mode": "replace",
          "script_score": {
            "script": {
              "inline": "binary_vector_score",
              "lang": "knn",
              "params": {
                "cosine": false,
                "field": "embedding_vector",
                "vector": face_vector
              }
            }
          }
        }
      },
      "size": 100
    })
    return res


__author__="Madhur_Dheer"

app=Flask(__name__,template_folder='templates')

App_root=os.path.dirname("maintype")

@app.route("/knn")
def classify():                                              ###CLASIFIER OPTION -A using KNN
    del1=pp.load(open('model_3.p','rb'))
    a_vector=vector(del1)
    print(a_vector)
    if (a_vector==0):
        return "No face detected"
    else:
        p_1=pp.load(open('model.p','rb'))
        p_2=pp.load(open('model_1.p','rb'))
        pred=p_1.predict([a_vector])
        pred1=list(p_2.inverse_transform(pred))
        return pred1[0]

@app.route("/cnn")
def conv():
    return "CNN"
@app.route("/ess")
def esearch():
    return "Elastic-search"

def vector(destination):                                  ###CONVERTING IMAGE INTO 128 vectors --DLIB
    predictor_path = "shape_predictor_5_face_landmarks.dat"
    face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
    faces_folder_path ="/home/siddharth/faceRecognition/src/"+destination
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)
    facerec = dlib.face_recognition_model_v1(face_rec_model_path)
    img = dlib.load_rgb_image(faces_folder_path)
    dets = detector(img, 1)
    print(len(dets))
    if (len(dets)==0):
        print("going to this parameter")
        return 0
    else:
        print("goinghere also")
        for k, d in enumerate(dets):
            shape = sp(img, d)
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            try_vector=face_descriptor
            return try_vector

@app.route("/")
def index():
    print("In Index")
    return render_template("upload.html")


@app.route("/upload",methods=['POST'])
def upload():
    print("In upload")
    target=os.path.join(App_root,"images/")

    if not os.path.isdir(target):
        os.mkdir(target)
    for file in request.files.getlist("file"):
        filename=file.filename
        destination="".join([target,filename])
        file.save(destination)
        vector(destination)
        pp.dump(destination,open('model_3.p','wb'))          ##loading the file
        return render_template("complete.html")

if __name__== "__main__":
    app.run(debug=True,port=5000,host='127.0.0.1')
