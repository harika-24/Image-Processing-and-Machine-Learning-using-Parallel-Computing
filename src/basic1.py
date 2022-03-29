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
import time
import multiprocessing
from flask import Flask, render_template, request
from PIL import Image
from elasticsearch import Elasticsearch
from tensorflow.python.keras._impl.keras.preprocessing.image import img_to_array
from twilio.rest import Client
from flask import Flask, render_template, request, url_for


app = Flask(__name__, template_folder='templates')

App_root=os.path.dirname("maintype")
@app.route("/knn")
def classify(try_vector):                                                #CLASIFIER OPTION -A using KNN
    start_time = time.time()
    print("in classifier======================================================")
    p_1=pp.load(open('model.p','rb'))
    p_2=pp.load(open('model_1.p','rb'))

    pred = p_1.predict([try_vector])
    v = p_2.inverse_transform(pred)
    print(p_2.inverse_transform(pred))
    print("My program took", time.time() - start_time, "to run")
    return v




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
            d = classify(try_vector)  #knn
            print(d)
            # if(d=="Akash Bhaiya"):
            #
            #     account_sid = 'AC48a2b57630cde3ad7acc662ea91cf5fd'
            #     auth_token = '101da4d773c821ed0c60d7f7dd17cb98'
            #     client = Client(account_sid, auth_token)
            #
            #     message = client.messages \
            #         .create(
            #         body="Employee Akash entered",
            #         from_='+15052786996',
            #         to='+918826748151'
            #     )
            #
            #     print(message.sid)
            # else:
            #     account_sid = 'AC48a2b57630cde3ad7acc662ea91cf5fd'
            #     auth_token = '101da4d773c821ed0c60d7f7dd17cb98'
            #     client = Client(account_sid, auth_token)
            #
            #     message = client.messages \
            #         .create(
            #         body="intruder detected",
            #         from_='+15052786996',
            #         to='+918826748151'
            #     )
            #
            #     print(message.sid)
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
        if( option == "KNN"):
            name1 = vector(destination,option)
            name1 = str(name1[0])
            print(name1, type(name1))

            f = open('helloworld.html', 'w')
            # name = "Akash Bhaiya"
            name = name1 + '.jpg'
            print(name)
            name2 = "/home/sethiamayank14/PycharmProjects/project2/src/images/"+ name
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









if __name__== "__main__":

    app.run(debug=True,port=5001,host='127.0.0.1')


