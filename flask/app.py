from flask import Flask, request, redirect, url_for
import torch
from PIL import Image

from utils import load_model, prepare_img, ID2NAME

app = Flask(__name__)

model = load_model()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_label(img):
    img = prepare_img(img)
    with torch.no_grad():
        output = model(img)[0]
    probs = torch.softmax(output, dim=0)
    top3_labels = []
    top3_probs = []
    for label_id in torch.topk(probs, 3).indices.numpy():
        top3_labels.append(ID2NAME[label_id])
        top3_probs.append(probs[label_id].item())
    return top3_labels, top3_probs


@app.route('/predict/<result>')
def predict(result):
    return result


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
            top3_labels, top3_probs = predict_label(img)
            result = dict()
            for i, (label, prob) in enumerate(zip(top3_labels, top3_probs)):
                if i != 0 and prob < 0.02:
                    break
                prob = int(prob * 100)
                result[label] = f'{prob}%'

            return redirect(url_for('predict', result=result))

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
    app.run(host='0.0.0.0', port=5001)
