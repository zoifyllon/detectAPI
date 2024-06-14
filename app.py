import numpy as np
from io import BytesIO
import urllib 
from flask import Flask, request, jsonify
from keras.preprocessing import image
from tensorflow.keras.applications import xception
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('Xception_multilabel_bangkit.h5')
list_labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'GrApple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot(Gray_leaf_spot)', 'Corn_(maize)___Common_rust', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

def predict(model, url):
    img_width, img_height = 96, 96
    with urllib.request.urlopen(url) as url:
        img = image.load_img(BytesIO(url.read()), target_size = (img_width, img_height, 3))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = xception.preprocess_input(img)
    return model.predict(img)

def predict_class(model, filename):
    prob_predict = predict(model, filename)[0]
    over_thr = np.where(prob_predict > 0.5)[0]
    results = {}
    for val in over_thr:
        key = list_labels[val].split('___')[-1].replace('_',' ')
        if key not in results:
            results[key] = prob_predict[val]
        else:
            results[key] = max(results[key], prob_predict[val])
    results = dict(sorted(results.items(), key = lambda x: x[1]))
    return results


@app.route('/predict', methods=['GET'])
def upload_file():
    if 'url' not in request.args:
        return jsonify({'error': 'No request'}), 400
    url = request.args['url']
    if 'https://storage.googleapis.com/zoifyllon-bucket/' not in url:
        return jsonify({'error': 'wrong url'}), 400
    result = predict_class(model, url)
    return {
        "message" : "success",
        "data" : {k:int(v*10000)/100 for k,v in result.items()}
        }, 200
    
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
