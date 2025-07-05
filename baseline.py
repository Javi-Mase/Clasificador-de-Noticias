#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# BASELINE basado en https://realpython.com/python-keras-text-classification/#defining-a-baseline-model
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from stop_words import get_stop_words

# lista de strings
spanishSW = get_stop_words("spanish")

# Comprobar argumentos de entrada
if len(sys.argv) != 3:
    sys.exit("Uso: python baseline.py <train.csv> <test.csv>")

trainPath, testPath = sys.argv[1], sys.argv[2]

# Cargar datos con pandas
trainDF = pd.read_csv(trainPath)
testDF  = pd.read_csv(testPath)

# Verificar que existen las columnas necesarias
for df, name in [(trainDF, "train"), (testDF, "val")]:
    if not {"text", "label"}.issubset(df.columns):
        sys.exit(f"Error: el archivo {name}.csv debe tener columnas 'text' y 'label'")

# Extraer arrays de texto y etiquetas
# array de strings
sentencesTrain = trainDF["text"].values
# array de 0/1
yTrain = trainDF["label"].values

sentencesTest = testDF["text"].values
yTest = testDF["label"].values

# Vectorización Bag-of-Words
# Creamos el vectorizador con parámetros por defecto
vectorizer = CountVectorizer(
    input="content",
    lowercase=True, # Para normalizar todo a minusculas y esta bien para palabras tipo ONG y ong
    stop_words=None,  # stop_words="spanishSW": usa la lista de palabras vacías de la biblioteca stop words para español. Aunque el profesor ha dicho de momento dejarlo asi, si eso luego probarlo con eso
    ngram_range=(1, 1), #(1,1) solo cuenta palabras, (2,2) cuenta bigramas, profesor: yo lo dejaria por defecto (1,1)
    max_df=1.0, # 0.8 descarta términos que aparecen en más del 80 % de los documentos (muy genéricos).Profesor: este valor no se suele cambiar se suele dejar al 100%
    min_df=0.05, #descarta términos que aparecen en menos de 5 documentos (ruido). Profesor: 5% es buen procentaje ya que tienes muchas noticias
    max_features=50000, #construye un vocabulario de tamaño fijo. Esto ponlo segun tengas problemas de memoria.  
)


# Se aprende el vocabulario SOLO del train
vectorizer.fit(sentencesTrain)

# Transformamos train y test a matrices dispersas
X_train = vectorizer.transform(sentencesTrain)
X_test  = vectorizer.transform(sentencesTest)

# Entrenar Regresión Logística
# max_iter alto para asegurar convergencia
classifier = LogisticRegression(max_iter=10000)
classifier.fit(X_train, yTrain)

# Guardar a disco vectorizer y clasificador
vect_path = "baselineVectorizer.joblib"
clf_path  = "baselineClassifier.joblib"
joblib.dump(vectorizer, vect_path)
joblib.dump(classifier,  clf_path)

# Evaluación sobre el conjunto de validación
y_pred = classifier.predict(X_test)

# Métricas
acc = accuracy_score(yTest, y_pred)
f1  = f1_score(yTest, y_pred, average="macro")

print("=== Baseline: Bag-of-Words + LogisticRegression ===")
print(f"Nº ejemplos train: {len(yTrain)}, vocabulario: {len(vectorizer.vocabulary_)}")
print(f"Nº ejemplos test : {len(yTest)}")
print(f"Accuracy (test): {acc:.4f}")
print(f"F1 macro   (test): {f1:.4f}")
