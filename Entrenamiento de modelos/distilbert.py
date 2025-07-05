#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# imports
import sys, torch, pandas as pd
from datasets import Dataset
from torch.utils.data import (WeightedRandomSampler, DataLoader)
from torch.nn import CrossEntropyLoss
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding)
from sklearn.metrics import accuracy_score, f1_score

# Revisamos argumentos
if len(sys.argv) != 3:
    sys.exit("Uso: python distilbert.py train.csv test.csv")

# Cargamos argumentos
trainPath, testPath = sys.argv[1:]

# leer CSV
trainDF = pd.read_csv(trainPath)
testDF   = pd.read_csv(testPath)

# Comprobamos que tiene todo
for df in (trainDF, testDF):
    if not {"text", "label"}.issubset(df.columns):
        sys.exit("CSV debe tener columnas 'text' y 'label'")

# Convertir a HuggingFace Dataset
trainDataSet = Dataset.from_pandas(trainDF)
testDataSet   = Dataset.from_pandas(testDF)

# tokenizar
modelName  = "distilbert-base-uncased"
tokenizer  = AutoTokenizer.from_pretrained(modelName)

# Dataset.map para tokenizar y truncar a 512
trainDataSet = trainDataSet.map(lambda ex: tokenizer(ex["text"], truncation=True), batched=True)
testDataSet   = testDataSet.map(lambda ex: tokenizer(ex["text"], truncation=True), batched=True)

# Asegurar etiquetas como int
trainDataSet = trainDataSet.map(lambda x: {"label": int(x["label"])})
testDataSet   = testDataSet.map(lambda x: {"label": int(x["label"])})

# Limitamos el formato para que el collator solo vea tensores
trainDataSet.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
testDataSet.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Cargar modelo binario
model = AutoModelForSequenceClassification.from_pretrained(
    modelName, num_labels=2,
    id2label={0: "human", 1: "ai"},
    label2id={"human": 0, "ai": 1}
)

# pesos de clase  (≈ 3 : 1)
N_total = len(trainDF)
N_h     = (trainDF["label"] == 0).sum()
N_ai    = (trainDF["label"] == 1).sum()
class_weights = torch.tensor([N_total / (2 * N_h), N_total / (2 * N_ai)], dtype=torch.float32)

# sampler balanceado: hace que cada batch llegue 50 / 50
sample_weights = [class_weights[0].item() if y == 0 else class_weights[1].item() for y in trainDF["label"]]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# Trainer personalizado: CrossEntropy ponderada + DataLoader con sampler balanceado
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")                                                  # 1) extraemos las etiquetas
        outputs = model(**inputs)                                                      # 2) pasamos el resto al modelo
        logits = outputs.logits                                                        # 3) recuperamos los logits sin normalizar
        weighted_loss = CrossEntropyLoss(weight=class_weights.to(logits.device))       # 4) instanciamos la pérdida con pesos
        loss = weighted_loss(logits, labels)                                           # 5) calculamos la pérdida ponderada
        return (loss, outputs) if return_outputs else loss                             # 6) devolvemos según lo esperado

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,                     # usamos el dataset de entrenamiento
            batch_size=self.args.train_batch_size,  # respetamos el batch size de los args
            sampler=sampler,                        # aplicamos el WeightedRandomSampler
            collate_fn=self.data_collator           # usamos el collator para padding dinámico
        )

# convertir logits → IDs antes de métricas (profesor)
def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels

# Cálculo de métricas: extraer el vector de predicciones si viene en tupla
def compute_metrics(p):
    preds = p.predictions
    if isinstance(preds, tuple): preds = preds[0]
    labels = p.label_ids
    if isinstance(labels, tuple): labels = labels[0]
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds, average="macro")}    

# Argumentos de entrenamiento
args = TrainingArguments(
    output_dir="./distilbert",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=6,
    learning_rate=2e-5,          # LR más bajo → más estable con pérdida ponderada
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
)

# Entrenamiento
WeightedTrainer(
    model=model,
    args=args,
    train_dataset=trainDataSet,
    eval_dataset=testDataSet,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    compute_metrics=compute_metrics,
).train()
