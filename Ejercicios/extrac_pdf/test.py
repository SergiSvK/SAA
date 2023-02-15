import json

# imprimir diccionario en formato json bonito

def print_json(dic):
    print(json.dumps(dic, indent=4))