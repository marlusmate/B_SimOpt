import json

with open('./metrics.json') as file:
    temp = json.load(file)

with open('../metrics.json', 'w+') as out:
    json.dump(temp, out)

#print(dix)
#print('Dateityp: ', type(dix))

