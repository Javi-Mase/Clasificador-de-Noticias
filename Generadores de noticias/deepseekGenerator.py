# -*- coding: utf-8 -*-  # Indicamos codificación del archivo


import json # Para poder manipular los json
import os  # Para manejar rutas de archivos y carpetas
import time  # Para esperar entre reintentos
import signal  # Para poder interrumpir el programa de manera segura (es decir hacer control c)
import argparse  # Para argumentos por línea de comandos
import requests  # Para hacer peticiones HTTP a la API

# CClase creada para poder definir funciones que nos permitan salir de manera segura dle programa
class GracefulExiter:
    def __init__(self):
        self.should_exit = False
        signal.signal(signal.SIGINT, self.exit)
        signal.signal(signal.SIGTERM, self.exit)

    def exit(self, signum, frame):
        print("\n Deteniendo el programa")
        self.should_exit = True

# Cliente que se conecta a la API de DeepSeek
class DeepSeekAPIClient:
    
    # Inicializamos
    def __init__(self, api_key):
        self.api_key = api_key  # Guarda la clave de API
        self.api_url = "https://api.deepseek.com/v1/chat/completions"  # URL de la API
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}  # Encabezados HTTP
        self.max_retries = 5  # Número de reintentos
        self.base_delay = 10  # Tiempo base entre reintentos
        self.timeout = 30  # Tiempo máximo de espera de respuesta

    # Método que hace la petición a la API con el prompt
    def callApi(self, prompt):
        for attempt in range(self.max_retries):
            try:
                # Envia la petición a la API de DeepSeek
                response = requests.post(self.api_url, headers=self.headers, json={"messages": [{"role": "user", "content": prompt}], "model": "deepseek-chat", "temperature": 0.7, "max_tokens": 500}, timeout=self.timeout)
                response.raise_for_status()  # Lanza excepción si el código no es 200
                return response.json()['choices'][0]['message']['content']  # Devuelve solo el texto generado
            
            # Si se pasa el tiempo de espera
            except requests.exceptions.Timeout:
                delay = self.base_delay * (2 ** attempt)
                print(f"\n Tiempo de espera agotado. \n Reintentando en {delay}s")
                time.sleep(delay)
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    print("\n Error: Verifica la API key")
                    return None
                print(f"\n Error: HTTP {e.response.status_code}: {e.response.text}")
                break  # Sal del bucle de reintentos
            
            except Exception as e:
                print(f"\n Error: {str(e)}")
                break

        print("\n Máximos reintentos alcanzados")
        return None  # Devuelve None si no se pudo obtener respuesta

    # Genera el párrafo basado en el título y contenido
    def generate(self, title, content):
        # Cuenta palabras del contenido original para que el parrafo generado tenga longitud similar
        wordCount = len(content.split())
        # Creamos el prompt que será enviado a la API
        prompt = (
            "Como redactor jefe de periódico, genera un párrafo que:\n"
            "1. Desarrolle objetivamente este titular: '{title}'\n"
            "2. Longitud aproximada: {wordCount}±15 palabras\n"
            "3. Estructura piramidal invertida\n"
            "4. Estilo periodístico profesional\n"
            "5. Importante que me des directamente el parrafo generado para que lo copie y pegue en mi articulo, no escribas nada mas que no sea el parrafo\n"
            "6. No utilices marcadores de relleno como “(insertar …)”, “[insertar …]” ni similares; si falta algún dato, escríbelo tú de forma verosímil o reformula la frase, pero nunca dejes huecos.\n"
            "Texto generado:\n"
        ).format(title=title, content=content, wordCount=wordCount)
        print(prompt)
        # Llamamos a la API con el prompt
        return self.callApi(prompt)

# Limpia las claves del JSON para que sea mas robusto
def keyCleaner(datos):
    return {k.strip().encode('utf-8').decode('utf-8', 'ignore').lower(): v for k, v in datos.items()}

# Función que recorre los archivos JSON, genera contenido y los actualiza
def jsonProcessor(api_key, root_dir="Noticias"):
    exiter = GracefulExiter()  # Instancia para poder interrumpir el programa
    client = DeepSeekAPIClient(api_key)  # Cliente para acceder a la API
    stats = {'procesados': 0, 'errores': 0, 'existentes': 0}  # Contadores

    print("Verificando conexión API")
    test_response = client.callApi("Test de conexión")  # Verifica si la API responde
    if not test_response:
        print("Error: Fallo de conexión con la API")
        return

    # Recorre todos los archivos en el directorio
    for root, _, files in os.walk(root_dir):
        if exiter.should_exit:
            break
            
        for file in files:
            if exiter.should_exit:
                break
              
            # Solo tratamos archivo .json  
            if not file.endswith('.json'):
                continue

            filePath = os.path.join(root, file)
            print(f"\n Procesando: {filePath}")
            
            try:
                with open(filePath, "r+", encoding='utf-8', errors='replace') as f:
                    try:
                        data = json.load(f)  # Carga los datos del archivo JSON
                        
                    except json.JSONDecodeError:
                        print("Error: Archivo JSON corrupto")
                        stats['errores'] += 1
                        continue
                    
                    # Limpia las claves del archivo JSON
                    dataCleaned = keyCleaner(data)
                    
                    # Comprobamos que la clave titutulo y contenido estan en el json
                    if 'title' not in dataCleaned or 'content' not in dataCleaned:
                        print(f"Error: Claves faltantes. Detectadas {list(data.keys())}")
                        stats['errores'] += 1
                        continue
                    
                    # Comprobación de si el archivo no ha sido procesado antes
                    if "deepseek" in data:
                        print("El archivo ya tiene contenido generado")
                        stats['existentes'] += 1
                        continue

                    # Genera el párrafo
                    generated = client.generate(dataCleaned['title'], dataCleaned['content'])
                    
                    if generated:
                        data["deepseek"] = generated.strip()  # Guarda el párrafo generado
                        f.seek(0)
                        json.dump(data, f, indent=2, ensure_ascii=False)  # Sobrescribe el JSON
                        f.truncate()
                        stats['procesados'] += 1
                        print("El párrafo ha sido generado exitosamente")
                    else:
                        stats['errores'] += 1
                        print("Error: Fallo en la generación")
            
            except Exception as e:
                print(f"Error: {str(e)}")
                stats['errores'] += 1


    # Imprimimos los resultados de la ejecución para saber en caso de que haya fallado donde y cuantas veces
    print("\n" + "═" * 50)
    print(f"Procesados: {stats['procesados']}")
    print(f"Existentes: {stats['existentes']}")
    print(f"Errores: {stats['errores']}")
    print("═" * 50)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Generador de noticias con DeepSeek')
    parser.add_argument('--apiKey', type=str, required=True, help='API Key de DeepSeek')
    parser.add_argument('--dir', type=str, default="Noticias", help='Directorio de archivos JSON')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"Error: Directorio no encontrado {args.dir}")
        exit(1)
        
    print("\n=== Iniciando procesamiento ===")
    jsonProcessor(args.api_key, args.dir)
