#!/usr/bin/env python3

# Script para dividir el conjunto de noticias en train y test

import pandas as pd
from sklearn.model_selection import train_test_split
import sys

# Cargamos .csv
csvPath = sys.argv[1]
file = pd.read_csv(csvPath, encoding='utf-8')

# Print para comprobar posibles errores
print(f"Total filas: {len(file)}")

# División
trainFile, testFile = train_test_split(
    file,
    test_size=0.2,            # 20% para text
    random_state=42,          # semilla para reproducibilidad
    shuffle=True,             # baraja antes de dividir ya que el .csv lo he creado por bloques de noticias (de una real acto seguido van sus tres generadas)
    stratify=file['label']    # mantiene proporción de label=0/1
)

# Guardamos
trainFile.to_csv("train.csv", index=False, encoding='utf-8')
testFile.to_csv("test.csv",   index=False, encoding='utf-8')

# Mas prints para comprobar errores, se pueden eliminar luego
print("No ha habido errores:")
print(f"train.csv: {len(trainFile)} filas ({len(trainFile)/len(file)*100:.1f}%)")
print(f"test.csv:   {len(testFile)} filas ({len(testFile)/len(file)*100:.1f}%)")
