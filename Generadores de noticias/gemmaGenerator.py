# -*- coding: utf-8 -*-
import os
import json
import time
import signal
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class GracefulExiter:
    def __init__(self):
        self.should_exit = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print("\n Deteniendo el programa")
        self.should_exit = True

class GemmaClient:
    def __init__(self):
        print("Cargando tokenizer desde carpeta local")
        self.tokenizer = AutoTokenizer.from_pretrained("/data/javiergarciam/modelos/gemma-3-1b-pt", legacy=False)
        
        print("Cargando modelo desde carpeta local")
        self.model = AutoModelForCausalLM.from_pretrained("/data/javiergarciam/modelos/gemma-3-1b-pt", torch_dtype=torch.bfloat16, device_map="auto")
        
        self.model.eval()

    def generate(self, title, content):
        wordCount = len(content.split())
        prompt = (
            "Como redactor jefe de periódico, genera un párrafo que:\n"
            "1. Desarrolle objetivamente este titular: '{title}'\n"
            "2. Longitud aproximada: {wordCount}±15 palabras\n"
            "3. Estructura piramidal invertida\n"
            "4. Estilo periodístico profesional\n"
            "5. Importante que me des directamente el parrafo generado para que lo copie y pegue en mi articulo, no escribas nada mas que no sea el parrafo\n"
            "6. No utilices marcadores de relleno como “(insertar …)”, “[insertar …]” ni similares; si falta algún dato, escríbelo tú de forma verosímil o reformula la frase, pero nunca dejes huecos.\n"
            "Texto generado:\n"
        )

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            generatedText = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return generatedText[len(prompt):].strip()
        except Exception as e:
            raise RuntimeError(str(e))

def keyCleaner(datos):
    return {k.strip().encode('utf-8').decode('utf-8', 'ignore').lower(): v for k, v in datos.items()}

def jsonProcessor(root_dir="Noticias"):
    exiter = GracefulExiter()
    client = GemmaClient()
    stats = {'procesados': 0, 'errores': 0, 'existentes': 0}

    for root, _, files in os.walk(root_dir):
        if exiter.should_exit:
            break
        
        for file in files:
            if exiter.should_exit:
                break
            
            if not file.endswith('.json'):
                continue

            file_path = os.path.join(root, file)
            print(f"\nProcesando: {file_path}")

            try:
                with open(file_path, "r+", encoding='utf-8', errors='replace') as f:
                    try:
                        datos = json.load(f)
                    except json.JSONDecodeError:
                        print("Error: Archivo JSON corrupto")
                        stats['errores'] += 1
                        continue

                    dataCleaned = keyCleaner(datos)

                    if 'title' not in dataCleaned or 'content' not in dataCleaned:
                        print(f"Error: Claves faltantes. Detectadas {list(datos.keys())}")
                        stats['errores'] += 1
                        continue

                    if "gemma" in datos:
                        print("El archivo ya tiene contenido generado")
                        stats['existentes'] += 1
                        continue

                    try:
                        generated = client.generate( dataCleaned['title'], dataCleaned['content'])
                        
                        if generated:
                            datos["gemma"] = generated
                            f.seek(0)
                            json.dump(datos, f, indent=2, ensure_ascii=False)
                            f.truncate()
                            print("El párrafo ha sido generado exitosamente")
                            stats['procesados'] += 1
                        else:
                            raise ValueError("Texto generado vacío")

                    except Exception as e:
                        print(f"Error:\n{e}")
                        stats['errores'] += 1

            except Exception as e:
                print(f"Error: {str(e)}")
                stats['errores'] += 1

    print("\n" + "═" * 50)
    print(f"Procesados: {stats['procesados']}")
    print(f"Existentes: {stats['existentes']}")
    print(f"Errores: {stats['errores']}")
    print("═" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generador de noticias con Gemma')
    parser.add_argument('--dir', type=str, default="Noticias", help='Directorio raíz de los archivos JSON')
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"Error: Directorio no encontrado {args.dir}")
        exit(1)

    print("\n=== Iniciando procesamiento con Gemma ===")
    jsonProcessor(args.dir)
