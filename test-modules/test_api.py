import requests
import time

url = "https://asr.hpda.vn/recog"

payload={'language': '1'}
files=[
('the_file', open('audio.wav','rb'))
]

headers = {}

start_time = time.time()

response = requests.request("POST", url, headers=headers, data=payload, files=files)

if response.ok:
    # print("abc")
    print(response)
else:
    print(response.text)

print("--- %s seconds ---" % (time.time() - start_time))