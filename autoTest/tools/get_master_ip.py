import json

str = ''
with open("/home/hostfile.json", 'r', encoding='utf-8') as f:
  temp = json.loads(f.read())
  for item in temp:
    if item['role'] == 'master':
        str = item['ip']
print(str)
