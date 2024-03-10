from flask import Flask, jsonify, request
import boto3
import json
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import joblib
from keras.preprocessing import image

predictor1 = joblib.load('cnn_model.pkl')
predictor2 = joblib.load('VG19_model_better.pkl')
app = Flask(__name__)

@app.route('/process_image', methods=['POST'])
def predict_on_s3_event():
    # Retrieve input data from the request
    data = request.get_json()

    # Extract necessary information from the request data
    image_key = data.get('image_key')
    aws_access_key_id = data.get('aws_access_key_id')
    aws_secret_access_key = data.get('aws_secret_access_key')
    s3_bucket_name = data.get('s3_bucket_name')
    for record in data['Records']:
        image_key = record['s3']['object']['key']

   
        region_name = 'us-east-1'
        s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
        response = s3.get_object(Bucket=s3_bucket_name, Key=image_key)
        image_bytes = response['Body'].read()

        img = Image.open(BytesIO(image_bytes))
        cv2.imwrite('image.jpg', np.array(img))

        image_1 = img.resize((64, 64))
        image_array = np.array(image_1)
        image_array = np.expand_dims(image_array, axis=0)
        result = predictor1.predict(image_array)

        if result[0][0] == 0:
            prediction = 'not infected'
        else:
            img = img.resize((224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            prediction = predictor2.predict(img_array).tolist()
    
        json_content = {"name": image_key, "prediction": prediction}

        s3.put_object(Body=json.dumps(json_content), Bucket=s3_bucket_name, Key=image_key)

        print(f'JSON file uploaded to {s3_bucket_name}/{image_key}')
        print(f'Prediction result for {image_key}: {response}')

    # The rest of your code remains the same...

    return jsonify({"message": "Prediction completed and results uploaded to S3."})

if __name__ == '__main__':
    app.run(debug=True)
