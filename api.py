from tensorflow import keras
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def recognize_image():
    img_path = request.get_json(force=True)['image_path']

    # prepare image for prediction
    img = keras.preprocessing.image.load_img(img_path, target_size=(180,180))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  
    
    # predict
    predictions = model.predict(img_array)
    if predictions[0] >= 0.5:
        label = "Dog"
        confidence = 100*(predictions[0])
    else:
        label = "Cat"
        confidence = 100*(1 - predictions[0])

    # prepare api response
    result = {
                "prediction" : [
                {
                "label" : label,
                "confidence" : confidence.tolist(),
                }
                ]
                }

    return jsonify(result)



if __name__ == '__main__':
    model = keras.models.load_model('saved_models/save_at_29.h5')
    app.run(debug=True, host='0.0.0.0', port=5000)