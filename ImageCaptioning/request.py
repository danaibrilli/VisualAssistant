import requests
headers = {
    'accept': 'application/json',
    'Content-Type': 'multipart/form-data',
}

files = {
    'image': ('./test_img.png;type', open('./test_img.png;type', 'rb')),
}

response = requests.post('http://max-image-caption-generator.codait-prod-41208c73af8fca213512856c7a09db52-0000.us-east.containers.appdomain.cloud/model/predict', headers=headers, files=files)
