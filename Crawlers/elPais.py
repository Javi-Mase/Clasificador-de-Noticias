# -*- coding: utf-8 -*-  # Especifica la codificación del archivo

# Importación de bibliotecas necesarias
import scrapy                   # Framework principal para web scraping
import json                     # Manipulación de datos JSON
import random                   # Generación de números aleatorios para nombres de archivo
import os                       # Operaciones del sistema para manejo de directorios
from bs4 import BeautifulSoup   # Parseo de HTML/XML

# Definición del spider para el sitio web de El País
class ElPaisSpider(scrapy.Spider):
    name = 'elpais'  # Nombre del spider para usarlo en consola con 'scrapy crawl elpais'
    allowed_domains = ['elpais.com']  # Limita el scraping solo a este dominio
    start_urls = ['https://elpais.com/']  # URL desde donde empieza el rastreo

    # Configuración personalizada para este spider
    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)...',  # Agente de usuario para simular navegador
        'LOG_ENABLED': False,  # Desactiva logs detallados
        'FEED_EXPORT_ENCODING': 'utf-8',  # Asegura codificación de salida
        'ROBOTSTXT_OBEY': False,  # Ignora robots.txt
        'DOWNLOAD_DELAY': 1.5  # Espera entre peticiones para no sobrecargar el sitio
    }

    def parse(self, response):  # Método que se ejecuta con cada respuesta
        url = response.url.strip()  # Guarda y limpia la URL actual

        # Extrae scripts que contienen datos estructurados en formato JSON-LD
        for script in response.css('script[type="application/ld+json"]'):
            try:
                data = json.loads(script.css('::text').get().strip())  # Intenta parsear el contenido JSON

                # Si hay varios objetos JSON-LD
                if isinstance(data, list):
                    for item in data:
                        if self.is_news_article(item):  # Verifica si es un artículo
                            yield from self.process_article(item, response, url)
                else:
                    if self.is_news_article(data):  # Si es solo un objeto JSON
                        yield from self.process_article(data, response, url)

            except Exception as e:
                self.logger.error("Error al parsear JSON: %s", e)  # Log de error
                continue

        # Recorre enlaces para seguir navegando por el sitio
        for next_page in response.css('a[href*="/"]::attr(href)').getall():
            if '/autor/' not in next_page and 'elpais.com' in next_page:  # Evita enlaces a autores
                yield response.follow(next_page, self.parse)  # Sigue el enlace llamando a parse

    # Comprueba si un bloque JSON representa un artículo de noticias
    def is_news_article(self, data):
        type_value = data.get('@type', [])
        return ('NewsArticle' in type_value) if isinstance(type_value, list) else (type_value == 'NewsArticle')

    def process_article(self, data, response, url):  # Procesa y guarda el artículo
        # Extrae campos básicos del JSON-LD
        title = data.get('headline', '')
        date = data.get('datePublished', '')[:10]  # Solo la fecha (sin hora)
        section = data.get('articleSection', 'general').lower().replace(' ', '_').replace('/', '_')[:30]  # Normaliza sección

        # Busca el contenido del artículo con BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        article_body = soup.find('div', {'data-dtm-region': 'articulo_cuerpo'})  # Contenedor principal
        content = ""

        if article_body:
            first_paragraph = article_body.find('p')  # Extrae el primer párrafo
            if first_paragraph:
                content = first_paragraph.get_text(separator=' ', strip=True)  # Texto limpio

        # Verifica que título y contenido existan y el contenido tenga cierta longitud mínima
        if not (title and content and len(content) > 50):
            return

        # Prepara los datos a guardar
        article_data = {
            'url': url,
            'title': title,
            'date': date,
            'section': section,
            'content': content
        }

        base_dir = 'elpais'  # Carpeta raíz para los artículos
        section_dir = os.path.join(base_dir, section)  # Subcarpeta por sección
        os.makedirs(section_dir, exist_ok=True)  # Crea la carpeta si no existe

        filename = os.path.join(section_dir, str(random.random()).replace(".", "") + ".json")  # Nombre aleatorio
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(article_data, f, indent=4, ensure_ascii=False)  # Guarda el artículo en JSON

        yield article_data  # Devuelve el resultado al pipeline de Scrapy
