import json

str = ''
with open("/home/hostfile.json", 'r', encoding='utf-8') as f:
  temp = json.loads(f.read())
  print(len(temp))

