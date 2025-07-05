# -*- coding: utf-8 -*-  # Define la codificación de caracteres del archivo como UTF-8
import json  # Importa el módulo JSON para leer y escribir archivos .json
import os  # Importa el módulo OS para interactuar con el sistema de archivos
import time  # Importa el módulo time para funciones relacionadas con el tiempo (no usado directamente aquí)
import signal  # Importa el módulo signal para manejar señales del sistema (SIGINT, SIGTERM)
import argparse  # Importa el módulo argparse para parsear argumentos de línea de comandos
from transformers import AutoTokenizer, AutoModelForCausalLM  # Importa clases de Hugging Face Transformers para tokenización y modelo causal
import torch  # Importa PyTorch para manejo de tensores y configuración del modelo

class GracefulExiter:
    def __init__(self):  # Constructor de la clase
        self.should_exit = False  # Flag que indica si debe terminar el bucle principal
        signal.signal(signal.SIGINT, self.exit_gracefully)  # Registra handler para Ctrl+C (SIGINT)
        signal.signal(signal.SIGTERM, self.exit_gracefully)  # Registra handler para terminación (SIGTERM)

    def exit_gracefully(self, signum, frame):  # Función llamada al recibir la señal
        print("\n Deteniendo el programa")  # Muestra mensaje de detención
        self.should_exit = True  # Marca el flag para salir del bucle principal

class Llama2Client:
    def __init__(self):  # Constructor que carga el tokenizer y el modelo
        print("Cargando tokenizer desde carpeta local")
        self.tokenizer = AutoTokenizer.from_pretrained("/data/javiergarciam/modelos/llama2-7b")  # Carga el tokenizer de LLaMA 2

        print("Cargando modelo desde carpeta local")
        self.model = AutoModelForCausalLM.from_pretrained("/data/javiergarciam/modelos/llama2-7b",
            torch_dtype=torch.float16,  # Indica usar half precision para menor consumo de memoria
            device_map="auto"           # Distribuye el modelo automáticamente en GPUs disponibles
        )

    def generate(self, title, content):  # Método que genera un párrafo a partir de título y contenido
        wordCount = len(content.split())  # Calcula número de palabras en el contenido

        # Construye el prompt para el modelo
        prompt = (
            "Como redactor jefe de periódico, genera un párrafo que:\n"
            "1. Desarrolle objetivamente este titular: '{title}'\n"
            "2. Longitud aproximada: {wordCount}±15 palabras\n"
            "3. Estructura piramidal invertida\n"
            "4. Estilo periodístico profesional\n"
            "5. Importante que me des directamente el parrafo generado para que lo copie y pegue en mi articulo, no escribas nada mas que no sea el parrafo\n"
            "6. No utilices marcadores de relleno como “(insertar …)”, “[insertar …]” ni similares; si falta algún dato, escríbelo tú de forma verosímil o reformula la frase, pero nunca dejes huecos.\n"
            "Texto generado:\n"
        ).format(title=title, content=content, wordCount=wordCount)  # Aplica formato con variables

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)  # Tokeniza el prompt y lo mueve al dispositivo del modelo
        outputs = self.model.generate(**inputs, max_new_tokens=500, do_sample=False)  # Genera texto con el modelo (determinístico)
        # Decodifica la salida, remueve tokens especiales y extrae solo el párrafo generado
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("Texto generado:")[-1].strip()

# Función para limpiar y normalizar las claves de un diccionario
# Devuelve un nuevo diccionario con claves en minúsculas y sin espacios extra
def keyCleaner(datos):  
    return {k.strip().encode('utf-8').decode('utf-8', 'ignore').lower(): v for k, v in datos.items()}

# Función principal para procesar todos los archivos JSON
def jsonProcessor(root_dir="Noticias"):  
    exiter = GracefulExiter()  # Instancia el manejador de terminación graciosa
    client = Llama2Client()  # Instancia el cliente de LLaMA 2
    stats = {'procesados': 0, 'errores': 0, 'existentes': 0}  # Diccionario para estadísticas

    # Recorre recursivamente el directorio de noticias
    for root, _, files in os.walk(root_dir):
        # Verifica si se ha solicitado salida
        if exiter.should_exit:  
            break
        
        for file in files:
            
            # Verifica nuevamente antes de procesar cada archivo
            if exiter.should_exit:
                break
            
            # Omite archivos que no terminen en .json
            if not file.endswith('.json'):  
                continue

            filePath = os.path.join(root, file)  # Construye la ruta completa
            print(f"\nProcesando: {filePath}")  # Imprime mensaje de inicio de proceso

            try:
                with open(filePath, "r+", encoding='utf-8', errors='replace') as f:  # Abre archivo en modo lectura-escritura
                    try:
                        data = json.load(f)  # Intenta cargar el JSON
                    except json.JSONDecodeError:
                        print("Error: Archivo JSON corrupto")
                        stats['errores'] += 1
                        continue

                    dataCleaned = keyCleaner(data)  # Normaliza claves

                    # Verifica que existan clave 'title' y 'content'
                    if 'title' not in dataCleaned or 'content' not in dataCleaned:
                        print(f"Error: Claves faltantes. Detectadas {list(data.keys())}")
                        stats['errores'] += 1
                        continue

                    # Salta si ya existe contenido generado bajo 'llama'
                    if "llama" in data:
                        print("El archivo ya tiene contenido generado")
                        stats['existentes'] += 1
                        continue

                    # Genera párrafo con LLaMA 2
                    generated = client.generate(dataCleaned['title'], dataCleaned['content'])

                    if generated:  # Si la generación fue exitosa
                        data["llama"] = generated.strip()  # Añade el párrafo al diccionario bajo la clave 'llama'
                        f.seek(0)  # Vuelve al inicio del archivo
                        json.dump(data, f, indent=2, ensure_ascii=False)  # Escribe el JSON modificado
                        f.truncate()  # Elimina cualquier contenido remanente tras la nueva escritura
                        stats['procesados'] += 1
                        print("El párrafo ha sido generado exitosamente")
                        
                    else:  # Si falló la generación
                        stats['errores'] += 1
                        print("Error: Fallo en la generación")

            except Exception as e:  # Captura errores críticos al abrir o procesar el archivo
                print(f"Error: {str(e)}")
                stats['errores'] += 1

    # Al finalizar, muestra resumen de estadísticas
    print("\n" + "═" * 50)
    print(f"Procesados: {stats['procesados']}")
    print(f"Existentes: {stats['existentes']}")
    print(f"Errores: {stats['errores']}")
    print("═" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generador con LLaMA 2')  # Crea parser de argumentos
    parser.add_argument('--dir', type=str, default="Noticias", help='Directorio de archivos JSON')  # Define argumento --dir
    args = parser.parse_args()  # Parsea argumentos de línea de comandos

    # Verifica que el directorio exista antes de procesar
    if not os.path.exists(args.dir):
        print(f"Error: Directorio no encontrado {args.dir}")
        exit(1)

    print("\n=== Iniciando procesamiento con LLaMA 2 7B ===")
    jsonProcessor(args.dir)
