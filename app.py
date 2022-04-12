
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import librosa
from tensorflow.keras.models import load_model
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def feature_extraction(file_name):
    y,sr=librosa.load(file_name)

    mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T,axis=0)
    melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40,fmax=8000).T,axis=0)
    chroma_stft=np.mean(librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=40).T,axis=0)
    chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr,n_chroma=40,bins_per_octave=40).T,axis=0)
    chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr,n_chroma=40,bins_per_octave=40).T,axis=0)

    features=np.reshape(np.vstack((mfccs,melspectrogram,chroma_stft,chroma_cq,chroma_cens)),(1,40,5))
    return features


app = Flask(__name__)


@app.route('/') 
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=='POST':
        file = request.files['audio']
        file.save(secure_filename(file.filename))
    
        # print(file)
        feature = feature_extraction(file.filename)
        # print(features)
        model = load_model('saved_models/audio_classification.hdf5')
        prediction = model.predict(feature)
        prediction = prediction.argmax(axis = -1)
        classes=['Air conditioner', 'Car horn', 'Children playing', 'Dog bark','Drilling',
                    'Engine idling', 'Gun shot', 'Jackhammer', 'Siren','Street music']

        for i ,cls in enumerate(classes):
            if prediction==i:
                predictn = cls

        # print(prediction)
        os.remove(file.filename)
        return render_template('pred.html',prediction=predictn)
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8888,debug=True)