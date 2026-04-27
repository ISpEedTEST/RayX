import requests

url = "http://localhost:8000/predict"
image_path = "C:\Users\khoba\Desktop\Medical_AI_V0.02\chest_xray\test\PNEUMONIA\123.jpeg"

with open(image_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print(response.json())