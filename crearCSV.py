#!/usr/bin/env python3

# Crea un csv con dos campos text y label recorriendo los .json
import sys
import json
import pathlib
import csv

# Campos que NO son texto de entrenamiento
SKIP_KEYS = {"url", "date", "section", "title",}

def process_folder(json_dir: pathlib.Path, out_csv: pathlib.Path):
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"No se encontraron .json en {json_dir}")
        return

    with out_csv.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv, quoting=csv.QUOTE_MINIMAL)
        # Cabecera
        writer.writerow(["title", "text", "label"])

        for js in json_files:
            data = json.loads(js.read_text(encoding="utf-8"))
            title = data.get("title", "").strip()
            # 1) Fila humana
            content = data.get("content", "").strip()
            if content:
                writer.writerow([title, content, 0])

            # 2) Filas IA: deepseek, llama, gemma, etc.
            for key, val in data.items():
                if key in SKIP_KEYS:
                    continue
                ia_text = (val or "").strip()
                if ia_text:
                    writer.writerow([title, ia_text, 1])

    print(f"✔️  CSV generado: {out_csv}  ({len(json_files)} archivos procesados)")

def main():
    if len(sys.argv) != 3:
        print("Uso: python 03_json_to_csv.py <carpeta_json> <salida.csv>")
        sys.exit(1)

    folder = pathlib.Path(sys.argv[1])
    out    = pathlib.Path(sys.argv[2])

    if not folder.is_dir():
        print(f"Error: {folder} no es un directorio válido")
        sys.exit(1)

    process_folder(folder, out)

if __name__ == "__main__":
    main()
