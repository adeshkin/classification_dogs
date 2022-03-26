import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import torch

from utils import load_model, allowed_file, prepare_img, UPLOAD_FOLDER


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model = load_model()


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('predict', filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/predict/<filename>')
def predict(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = prepare_img(filepath)

    output = model(img)
    _, pred_label = torch.max(output, 1)
    return f'Predicted label = {pred_label.numpy()[0]}'


if __name__ == "__main__":
    app.run()
