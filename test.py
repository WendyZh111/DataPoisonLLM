# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import base64
import requests

# OpenAI API Key
api_key = ""


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# Path to your image
image_path_1 = "data/image1.png"
image_path_2 = "data/image2.jpg"
image_path_3 = "data/image3.jpg"

# Getting the base64 string
# base64_image_1 = encode_image(image_path_1)
# base64_image_2 = encode_image(image_path_2)
# base64_image_3 = encode_image(image_path_3)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "gpt-4-turbo",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": ""
                },

                # {
                #     "type": "image_url",
                #     "image_url": {
                #         "url": f"data:image/jpeg;base64,{base64_image_3}"
                #     }
                # },
            ]
        }
    ],
    "max_tokens": 2048
}

# response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
response = requests.post("https://aigptx.top/v1/chat/completions", headers=headers, json=payload)
print(response.json()["choices"][0]["message"]["content"])
