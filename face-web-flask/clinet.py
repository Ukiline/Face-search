import requests

user_info = {'name': 'letian', 'password': '123'}
r = requests.post("http://10.16.88.110:5000/sketch", data=user_info)
print(r)
print(r.text)