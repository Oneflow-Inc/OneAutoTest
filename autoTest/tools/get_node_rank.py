import json
import socket

hostname = socket.gethostname()
ip = socket.gethostbyname(hostname)


i = 0
with open("/home/hostfile.json", 'r', encoding='utf-8') as f:
  temp = json.loads(f.read())
  for item in temp:
    if item['ip'] == ip:
      break
    else:
      i += 1
print(i)

