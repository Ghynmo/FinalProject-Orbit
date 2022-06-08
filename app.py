import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from keras.models import load_model
import numpy as np
from keras import utils


app = Flask(__name__)

model_path = 'modelcvorbit.h5'
model = load_model(model_path)

def model_predict(path, model):
    
    img = utils.load_img(path, target_size=(200, 200))
    x = utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    predict = model.predict(images)
    classes = np.argmax(predict)

    output = ""

    if classes == 0:
        output =  "ini adalah Gol 1"
    elif classes == 1:
        output =  "ini adalah Gol 2"
    elif classes == 2:
        output =  "ini adalah Gol 3"
    elif classes == 3:
        output =  "ini adalah Gol 4"
    elif classes == 4:
        output =  "ini adalah Gol 5"
    
    return output

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        os.remove(file_path)
        return preds
    return None

if __name__ == '__main__':
    app.run(debug=True)

