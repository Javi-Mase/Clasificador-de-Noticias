#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Compara múltiples modelos binarios "humano vs IA" usando un único conjunto de prueba e imprime para cada uno el informe detallado de clasificación (precision, recall, F1, soporte).
# Uso: python comparar_modelos.py --csv test.csv

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import argparse
import sys
import time
import joblib
import pandas as pd
import torch
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Nuestra lista de modelos
MODEL_DIRS = {
    "ALBETO"    : "./albeto/checkpoint-64464",
    "MARIA"     : "./maria/checkpoint-42976",
    "TWHIN"     : "./twhin/checkpoint-53720",
    "BETO"      : "./beto/checkpoint-32232",
    "DISTILBETO": "./distilbeto/checkpoint-32232",
    "DISTILBERT": "./distilbert/checkpoint-53720",
    "BERTIN"    : "./bertin/checkpoint-64464",
    "MDEBERTA"  : "./mdeberta/checkpoint-21488",
}



# Realiza la inferencia de un modelo dado sobre una lista de textos.
#  model_dir: ruta al directorio del checkpoint.
#  texts: lista de cadenas de entrada.
#  device: 'cuda' o 'cpu'. Para asi ejecutarlo en mi ordenado o no
#  batch_size: número de ejemplos procesados por paso.
#  Devuelve una lista de predicciones (0 o 1).
def inferir(model_dir: str, texts: list, device: str, batch_size: int = 32) -> list:
    # Cargar el tokenizer y el modelo entrenado desde el checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    # Mover el modelo al dispositivo (GPU o CPU) y ponerlo en modo evaluación
    model.to(device)
    model.eval()

    all_preds = []  # Aquí almacenaremos las predicciones finales

    raw_max = getattr(tokenizer, "model_max_length", None)
    # si el tokenizador no lo define o lo define como un número irreal,
    # forzamos un límite de 512
    max_len = raw_max if (isinstance(raw_max, int) and 1 <= raw_max <= 4096) else 512
    
    # Procesar los textos en batches para eficiencia
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]

        # Tokenizar y convertir a tensores PyTorch
        enc = tokenizer(
            batch,
            padding=True,        # añade padding para igualar longitudes
            truncation=True,     # trunca textos largos a max_length
            max_length=max_len,
            return_tensors="pt"  # devuelve tensores
        )
        
        # Mover tensores al mismo dispositivo que el modelo
        enc = {k: v.to(device) for k, v in enc.items()}

        # Calcular logits sin gradiente
        with torch.no_grad():
            logits = model(**enc).logits

        # Transformar logits a etiquetas (argmax) y añadir a la lista
        batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()
        all_preds.extend(batch_preds)

    return all_preds

# Carga el vectorizer y el clasificador guardados con joblib,
# transforma los texts con CountVectorizer y predice con LogisticRegression.
def inferir_baseline(texts: list) -> list:
    vec = joblib.load("baselineVectorizer.joblib")
    clf = joblib.load("baselineClassifier.joblib")
    X = vec.transform(texts)
    return clf.predict(X).tolist()



def main():
    parser = argparse.ArgumentParser(description="Compararador usando clasification report")
    
    parser.add_argument("--csv", "-c", required=True, help="Ruta al CSV de test (debe tener columnas 'text','label')")
    
    args = parser.parse_args()

    # Cargar y validar el conjunto de prueba
    df = pd.read_csv(args.csv)
    if not {"text", "label"}.issubset(df.columns):
        sys.exit("Error: el CSV debe contener las columnas 'text' y 'label'")

    # Extraer listas de textos y etiquetas verdaderas
    texts  = df["text"].astype(str).tolist()
    yTrue = df["label"].astype(int).tolist()

    # Determinar dispositivo de inferencia
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo para inferencia: {device}\n")

    # Para cada modelo, hacer inferencia y mostrar el reporte
    for alias, checkpointPath in MODEL_DIRS.items():
        print(f"\n=== Modelo: {alias} ({checkpointPath}) ===")
        
        model_for_count = AutoModelForSequenceClassification.from_pretrained(checkpointPath)
        total_params = sum(p.numel() for p in model_for_count.parameters())
        print(f"Parámetros: {total_params/1e6:.2f} M")
        del model_for_count
        torch.cuda.empty_cache()

        # Obtener predicciones
        t0 = time.time()
        yPred = inferir(checkpointPath, texts, device)
        dt = time.time() - t0
        print(f"Tiempo inferencia: {dt:.2f} s")

        # Generar y mostrar el classification_report
        report = classification_report(
            yTrue,
            yPred,
            target_names=["human", "IA"],
            digits=4,         # 4 decimales en la salida
            zero_division=0   # evita errores si alguna clase no predicha
        )
        print(report)
        
        # ---------- MATRIZ DE CONFUSIÓN ----------
        cm = confusion_matrix(yTrue, yPred, labels=[0, 1])

        # 2.1  Imprimirla en texto (opcional)
        print("Matriz de confusión (VN, FP / FN, VP):\n", cm, "\n")

        # 2.2  Dibujar y guardar como PNG (opcional)
        fig, ax = plt.subplots(figsize=(4,4))
        disp = ConfusionMatrixDisplay(
                 confusion_matrix=cm,
                 display_labels=["human", "IA"])
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"Confusión – {alias}")
        plt.tight_layout()
        fig.savefig(f"cm_{alias}.png", dpi=300)
        plt.close(fig)

        
        
    print("\n=== Baseline Bag-of-Words + LogisticRegression ===")
    y_pred_base = inferir_baseline(texts)
    print(classification_report(
        yTrue, y_pred_base,
        target_names=["human","IA"],
        digits=4, zero_division=0
    ))

if __name__ == "__main__":
    main()
