# -*- coding: utf-8 -*-
import scrapy
import json
import random
import os
from bs4 import BeautifulSoup
from scrapy.crawler import CrawlerProcess
from urllib.parse import urljoin, urlparse
from w3lib.url import safe_url_string

class ClarinSpider(scrapy.Spider):
    name = 'clarin'
    allowed_domains = ['www.clarin.com']  # Dominio específico
    start_urls = ['https://www.clarin.com/']  # URL de noticias actuales

    custom_settings = {
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'LOG_LEVEL': 'INFO',
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_DELAY': 2,
        'CONCURRENT_REQUESTS': 2,
        'HTTPCACHE_ENABLED': False,  # Desactivar caché temporalmente
        'RETRY_TIMES': 8,
        'FEED_EXPORT_ENCODING': 'utf-8'
    }

    def start_requests(self):
        # Forzar nueva descarga ignorando caché
        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse, meta={'dont_merge_cookies': True})

    def parse(self, response):
        # Selectores actualizados para nueva estructura
        articles = response.css('article a::attr(href), .story-link::attr(href)').getall()
        
        for link in articles:
            try:
                absolute_url = urljoin(response.url, safe_url_string(link))
                if any(kw in absolute_url for kw in ['/noticias/', '/article/', '/politica/', '/economia/']):
                    yield response.follow(absolute_url, self.parse_article)
            except Exception as e:
                self.logger.error(f"Error en URL: {link} - {str(e)}")

        # Paginación dinámica
        next_page = response.css('a[rel="next"]::attr(href), .pagination-next::attr(href)').get()
        if next_page:
            yield response.follow(next_page, self.parse)

    def parse_article(self, response):
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extracción mejorada
        title = soup.find('h1', {'itemprop': 'headline'}) or soup.find('h1')
        title = title.get_text(strip=True) if title else None
        
        date = soup.find('meta', {'property': 'article:published_time'}) or soup.find('time')
        date = date['content'][:10] if date and 'content' in date.attrs else date['datetime'][:10] if date else None
        
        content = ' '.join([p.get_text(strip=True) for p in soup.select('div.article-body p, article p')])
        
        section = urlparse(response.url).path.split('/')[1] if len(urlparse(response.url).path.split('/')) > 2 else 'general'

        if title and len(content) > 100:
            self.save_article(response.url, title, date, section, content)
            yield {
                'url': response.url,
                'title': title,
                'date': date,
                'section': section,
                'content': content
            }

    def save_article(self, url, title, date, section, content):
        base_dir = 'clarin_articles'
        section_dir = os.path.join(base_dir, section)
        os.makedirs(section_dir, exist_ok=True)
        
        filename = os.path.join(section_dir, f"{random.getrandbits(64)}.json")
        
        data = {
            'url': url,
            'title': title,
            'date': date,
            'section': section,
            'content': content
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n Artículo guardado en: {filename}")
        print(json.dumps(data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    process = CrawlerProcess()
    process.crawl(ClarinSpider)
    print("Iniciando crawler")
    process.start()
