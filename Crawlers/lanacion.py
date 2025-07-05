# -*- coding: utf-8 -*-
import scrapy
import json
import random
import os
from bs4 import BeautifulSoup
from scrapy.crawler import CrawlerProcess
from urllib.parse import urljoin, urlparse
from w3lib.url import safe_url_string

class lanacionSpider(scrapy.Spider):
    name = 'lanacion'
    allowed_domains = ['www.lanacion.com.ar']
    start_urls = ['https://www.lanacion.com.ar/']

    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'LOG_LEVEL': 'INFO',
        'FEED_EXPORT_ENCODING': 'utf-8',
        'DOWNLOAD_DELAY': 2,
        'CONCURRENT_REQUESTS': 1,
        'RETRY_TIMES': 2
    }

    def parse(self, response):
        # Seguimiento mejorado de enlaces
        for link in response.css('a::attr(href)').getall():
            try:
                absolute_url = urljoin(response.url, safe_url_string(link))
                if self.is_valid_url(absolute_url):
                    yield response.follow(absolute_url, self.parse)
            except Exception as e:
                self.logger.error(f"Error en enlace: {link} - {str(e)}")

        # Procesar datos estructurados
        for script in response.css('script[type="application/ld+json"]'):
            try:
                data = json.loads(script.css('::text').get().strip())
                if isinstance(data, list):
                    for item in data:
                        if self.is_news_article(item):
                            yield from self.process_article(item, response, response.url)
                else:
                    if self.is_news_article(data):
                        yield from self.process_article(data, response, response.url)
            except Exception as e:
                self.logger.error(f"Error en JSON-LD: {str(e)}")

    def is_valid_url(self, url):
        """Filtra URLs válidas para artículos"""
        parsed = urlparse(url)
        return (
            'lanacion.com.ar' in parsed.netloc and
            '/autor/' not in url and
            any(kw in url for kw in ['/politica/', '/economia/', '/sociedad/', '/cultura/', '/deportes/'])
        )

    def is_news_article(self, data):
        article_type = data.get('@type', [])
        return 'NewsArticle' in article_type if isinstance(article_type, list) else article_type == 'NewsArticle'

    def process_article(self, data, response, url):
        # Extracción de metadatos
        title = data.get('headline', '')
        date = data.get('datePublished', '')[:10]
        section = self.extract_section(url)
        
        # Extracción de contenido actualizada
        soup = BeautifulSoup(response.text, 'html.parser')
        content = self.extract_content(soup)
        
        # Validación mejorada
        if not self.valid_article(title, content):
            self.logger.warning(f"Artículo inválido: {url}")
            return

        # Guardado y visualización
        self.save_article(url, title, date, section, content)
        yield {
            'url': url,
            'title': title,
            'date': date,
            'section': section,
            'content': content
        }

    def extract_section(self, url):
        """Extrae la sección de la URL con manejo de errores"""
        try:
            path_parts = urlparse(url).path.split('/')
            return path_parts[1].lower().replace('-', '_') if len(path_parts) > 2 else 'general'
        except:
            return 'general'

    def extract_content(self, soup):
        """Nuevos selectores para contenido actualizados el 2024"""
        # Intentar múltiples estrategias de extracción
        content = ""
        
        # Estrategia 1: Buscar contenedor principal
        body = soup.find('div', class_='article-body') or \
              soup.find('article', class_='article-main') or \
              soup.find('div', {'data-article-body': True})
        
        # Estrategia 2: Buscar primer párrafo con texto sustancial
        if body:
            for p in body.find_all('p'):
                text = p.get_text(strip=True)
                if len(text) > 50 and not any(kw in text.lower() for kw in ['publicidad', 'seguí leyendo']):
                    content = text
                    break
        else:
            # Fallback: Buscar cualquier párrafo largo
            paragraphs = [p.get_text(strip=True) for p in soup.find_all('p') if len(p.get_text(strip=True)) > 100]
            content = paragraphs[0] if paragraphs else ""
        
        return content

    def valid_article(self, title, content):
        return len(title) > 15 and len(content) > 80

    def save_article(self, url, title, date, section, content):
        # Crear directorios
        base_dir = 'lanacion_articles'
        section_dir = os.path.join(base_dir, section)
        os.makedirs(section_dir, exist_ok=True)
        
        # Generar nombre de archivo
        filename = os.path.join(section_dir, f"{random.getrandbits(64)}.json")
        
        # Crear y guardar datos
        article_data = {
            'url': url,
            'title': title,
            'date': date,
            'section': section,
            'content': content
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(article_data, f, indent=2, ensure_ascii=False)
        
        # Mostrar preview detallado
        print("\n" + "="*70)
        print(f"🔥 NUEVO ARTÍCULO [{section.upper()}]")
        print(f"📅 {date} | {url}")
        print("-"*70)
        print(f"📰 Título: {title}")
        print(f"📝 Contenido: {content[:200]}...")
        print("="*70 + "\n")

if __name__ == "__main__":
    process = CrawlerProcess()
    process.crawl(lanacionSpider)
    print("🚀 Spider de La Nación iniciado - Presiona Ctrl+C para detener")
    process.start()