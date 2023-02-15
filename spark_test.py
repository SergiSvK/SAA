# multiple eleccion de subida de archivo en google colab

from google.colab import files
from google.colab import drive
from pyspark import SparkConf, SparkContext
import os.path
import re

conf = SparkConf()
conf.setMaster("local[1]")

conf.setAppName("MyAppName")
sc = SparkContext.getOrCreate(conf=conf)
print(sc.version)

# selección de archivo en google colab
local = input("Ingrese la forma de lectura del archivo: [l/n] ")
file_name = ""

if local == "l":
    # selección de archivo en google colab drive
    drive.mount('/content/drive')
    file_name = "/content/drive/MyDrive/Datasets/LoremIpsum.txt"
elif local == "n":
    # selección de archivo en google colab
    uploaded = files.upload()
    file_name = os.path.join('LoremIpsum.txt')
else:
    print("Opción inválida. Por favor, ingrese 'l' o 'n'.")


def remove_punctuation(texto):
    return re.sub(r'[^\w\s]', '', texto).lower().strip()


loremRDD = sc.textFile(file_name, 8).map(remove_punctuation).filter(lambda x: len(x) > 0)
loremRDD.take(10)

# Dos problemas del formato RDD
# El primer problema es que necesitamos obtener las palabras que contiene cada línea.

# Obtener las palabras separadas por comas

loremWordsRDD = loremRDD.flatMap(lambda x: x.split(","))

#mejorar de la efiencia de la memoria

loremWordsRDD.count()

loremWordsRDD.persist()

loremWordsRDD.unpersist()

# contar las palabras distintas usando el takeOrdered

top_15_words_and_counts = word_count(loremWordsRDD).takeOrdered()

# calcular algunas estadísticas

# Different words with exactly two 's'

loremWordsRDD.filter(lambda x: x.count("s") == 2).distinct().count()

# palabra que más se repite de siete letras

loremWordsRDD.filter(lambda x: len(x) == 7).map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y).takeOrdered(1, key=lambda x: -x[1])

# more consonants than vowels all words count

loremWordsRDD.filter(lambda x: len(x) > 0).map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y).filter(lambda x: sum([1 for c in x[0] if c in "aeiou"]) < sum([1 for c in x[0] if c in "bcdfghjklmnpqrstvwxyz"])).count()