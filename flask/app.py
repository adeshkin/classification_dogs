from flask import Flask, request, redirect, url_for
from PIL import Image
import numpy as np
import torch
from utils import allowed_file, load_model, prepare_img, ID2NAME

app = Flask(__name__)

model = load_model()


def predict(img):
    img = prepare_img(np.array(img))
    output = model(img)
    _, pred_label = torch.max(output, 1)
    idx = pred_label.numpy()[0]
    name = ID2NAME[idx]

    return name


@app.route('/result/<name>')
def show_result(name):
    return name


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(url_for('upload_file'))

        file = request.files['file']
        if file.filename == '':
            return redirect(url_for('upload_file'))

        if file and allowed_file(file.filename):
            img = Image.open(file.stream)
            label_name = predict(img)
            print(label_name)
            return redirect(url_for('show_result', name=label_name))

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


if __name__ == "__main__":
    app.run()
