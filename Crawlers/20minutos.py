# -*- coding: utf-8 -*-
import scrapy
import json
import random
import os
from bs4 import BeautifulSoup
from datetime import datetime

class VeinteMinutosSpider(scrapy.Spider):
    name = '20minutos'
    allowed_domains = ['20minutos.es']
    start_urls = ['https://www.20minutos.es/']

    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'LOG_LEVEL': 'WARNING',
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_DELAY': 2,
        'CONCURRENT_REQUESTS': 1
    }

    def parse(self, response):
        # Extraer artículos principales
        for article in response.css('article a::attr(href)').getall():
            if '/noticia/' in article:
                yield response.follow(article, self.parse_article)
        
        # Paginación
        for next_page in response.css('a[rel="next"]::attr(href)').getall():
            yield response.follow(next_page, self.parse)

    def parse_article(self, response):
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extracción de datos
        title = soup.find('h1', {'class': 'article-title'})
        title = title.text.strip() if title else None
        
        date = soup.find('time', {'itemprop': 'datePublished'})
        date = date['datetime'][:10] if date else None
        
        content = []
        body = soup.find('div', {'class': 'article-text'})
        if body:
            for p in body.find_all('p', class_='paragraph'):
                content.append(p.get_text().strip())
        content = ' '.join(content)
        
        section = response.url.split('/')[3] if len(response.url.split('/')) > 4 else 'general'
        section = section.lower().replace('-', '_')[:25]

        # Validación y logging
        if title and content and len(content) > 100:
            self.logger.warning("\n" + "-"*50)
            self.logger.warning(f"URL: {response.url}")
            self.logger.warning(f"Título: {title}")
            self.logger.warning(f"Sección: {section}")
            self.logger.warning("-"*50 + "\n")
            
            self.save_article(response.url, title, date, section, content)
            yield {
                'url': response.url,
                'title': title,
                'date': date,
                'section': section,
                'content': content
            }

    def save_article(self, url, title, date, section, content):
        base_dir = '20minutos'
        section_dir = os.path.join(base_dir, section)
        os.makedirs(section_dir, exist_ok=True)
        
        filename = os.path.join(section_dir, f"{random.getrandbits(64)}.json")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'url': url,
                'title': title,
                'date': date,
                'section': section,
                'content': content
            }, f, indent=2, ensure_ascii=False)

# Bloque de ejecución directa
if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess
    
    process = CrawlerProcess(settings={
        'LOG_LEVEL': 'WARNING',
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    })
    
    if not os.path.exists('20minutos'):
        os.makedirs('20minutos')
    
    print("🚀 Iniciando crawler de 20minutos...")
    process.crawl(VeinteMinutosSpider)
    process.start()
    print("✅ Proceso completado!")