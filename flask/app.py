from flask import Flask, request, redirect, url_for
from PIL import Image

from utils import allowed_file, predict


app = Flask(__name__)


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

            return redirect(url_for('upload_file'))

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
