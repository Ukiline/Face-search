from socket import socket
from flask import Flask, request
from PIL import Image
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from matplotlib.pyplot import eventplot
import json
import numpy as np
import json
import cv2
import base64

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
CORS(app, resources=r'/*')


# base64编码转换为图片
def base64_to_image(base64_data, file_name):
    # base64编码转换为图片
    # 去掉base64编码头部
    print(base64_data)
    base64_data = base64_data[base64_data.find(',') + 1:]
    base64_data = base64_data.replace(' ', '+') # 替换空格
    base64_data = base64_data.encode('utf-8') # 解码
    base64_data = base64.b64decode(base64_data) # 解码
    with open(file_name, 'wb') as f:
        f.write(base64_data)

# 图片中心裁剪
def crop_center():
    img = Image.open('./sketch.png')
    width, height = img.size
    tmp = 0
    if width < height:
        tmp = width / 2
    else:
        tmp = height / 2
    
    left = width / 2 - tmp
    top = height / 2 - tmp
    right = width / 2 + tmp
    bottom = height / 2 + tmp
    img = img.crop((left, top, right, bottom))
    # 保存裁剪后的图片
    img.save('./sketch.png')
    print(img.size)

# 图片转base64
def image_to_base64(file_name):
    with open(file_name, 'rb') as f:
        base64_data = base64.b64encode(f.read())
    return base64_data

@app.route('/')
def index():
    return '<h1>Hello World!</h1>'

@app.route('/postPic', methods=['POST'])
def postPic():
    # print(request.headers)
    # 打印body中的数据
    data = request.get_data()
    # json转换为字典
    data = json.loads(data)
    data = data['str']
    # base64编码转换为图片
    base64_to_image(data, './sketch.png')
    # 图片中心裁剪
    crop_center()
    # 图片转base64
    base64_data = image_to_base64('./sketch.png')
    # 转换为字符串
    base64_data = 'data:image/png;base64,' + base64_data.decode('utf-8')
    # 通过websocket发送数据
    event_name = "painting"
    socketio.emit(event_name, base64_data)
    return "ok"

@socketio.on('connect')
def connect():
    print("connect")

@socketio.on('disconnect')
def disconnect():
    print('disconnect')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=4000, debug=False)