# -*- coding: utf-8 -*-  # Define la codificación del archivo como UTF-8

# Importación de bibliotecas necesarias
import scrapy  # Framework de scraping
import json  # Para trabajar con estructuras JSON
import random  # Para generar nombres de archivo aleatorios
import os  # Para manipular rutas y carpetas
from bs4 import BeautifulSoup  # Para procesar el HTML de forma eficiente

# Definición del spider de Scrapy para el sitio web de El Mundo
class ElMundoSpider(scrapy.Spider):
    name = 'elmundo'  # Nombre del spider para ejecutarlo con scrapy crawl elmundo
    allowed_domains = ['elmundo.es']  # Restringe el rastreo solo a este dominio
    start_urls = ['https://www.elmundo.es/']  # URL desde la que comienza el rastreo

    # Configuraciones específicas para este spider
    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; WOW64)...',  # Agente para simular un navegador
        'FEED_EXPORT_ENCODING': 'utf-8',  # Exportación con codificación UTF-8
    }

    # Método principal que se ejecuta para cada respuesta recibida
    def parse(self, response):
        url = response.url.strip()  # Guarda la URL actual limpia

        # Extrae todos los bloques JSON-LD embebidos en la página
        for script in response.css('script[type="application/ld+json"]'):
            try:
                data = json.loads(script.css('::text').get())  # Intenta cargar el JSON

                # Si el JSON es una lista, itera sobre los elementos
                if isinstance(data, list):
                    for item in data:
                        if self.is_news_article(item):  # Si es un artículo, procesarlo
                            yield from self.process_article(item, response, url)
                else:
                    if self.is_news_article(data):  # Si es un único artículo, procesarlo
                        yield from self.process_article(data, response, url)

            except Exception as e:
                self.logger.error("Error al parsear JSON: %s", e)  # Log de error
                continue

        # Recolecta y sigue todos los enlaces encontrados en la página
        for next_page in response.css('a::attr(href)').getall():
            if next_page and 'elmundo.es' in next_page:  # Solo sigue enlaces del dominio
                yield response.follow(next_page, self.parse)  # Llama recursivamente a parse

    # Verifica si el JSON corresponde a un artículo de noticias
    def is_news_article(self, data):
        return data.get('@type') == 'NewsArticle'

    # Procesa los datos del artículo y los guarda como JSON
    def process_article(self, data, response, url):
        title = data.get('headline', '')  # Título del artículo
        date = data.get('datePublished', '')[:10]  # Fecha (solo YYYY-MM-DD)
        section = data.get('articleSection', 'actualidad').lower()  # Sección del artículo

        # Procesamiento del HTML con BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        article_paragraph = soup.find("p", class_="ue-c-article__paragraph")  # Primer párrafo del contenido

        if not article_paragraph:  # Si no se encuentra el párrafo, salir
            return

        content = article_paragraph.get_text(separator=" ").strip()  # Extrae el texto plano

        if not (title and content):  # Si falta título o contenido, omitir
            return

        # Registro en el log de la terminal
        self.logger.info("-------------------------")
        self.logger.info("URL: %s", url)
        self.logger.info("Título: %s", title)
        self.logger.info("Contenido: %s", content)
        self.logger.info("Fecha: %s", date)
        self.logger.info("Sección: %s", section)
        self.logger.info("-------------------------")

        # Guarda los datos en un diccionario
        article_data = {
            'url': url,
            'title': title,
            'date': date,
            'section': section,
            'content': content
        }

        base_dir = 'elmundo'  # Carpeta raíz donde se guardarán los artículos
        section_dir = os.path.join(base_dir, section)  # Subcarpeta por sección
        os.makedirs(section_dir, exist_ok=True)  # Crea la carpeta si no existe

        # Genera un nombre de archivo único
        filename = os.path.join(section_dir, str(random.random()).replace(".", "") + ".json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(article_data, f, indent=4, ensure_ascii=False)  # Guarda el contenido en formato JSON

        yield article_data  # Devuelve el diccionario (puede ser capturado por scrapy si se usa FEED)

