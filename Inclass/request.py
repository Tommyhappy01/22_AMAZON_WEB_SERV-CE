import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'TV':100, 'radio':200, 'newspaper':400})

print(r.json())