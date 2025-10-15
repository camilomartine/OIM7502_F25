import scrapy
from sp500_scraper.items import Sp500ScraperItem

class Sp500Spider(scrapy.Spider):
    name = "sp500"
    allowed_domains = ["slickcharts.com"]
    start_urls = ["https://www.slickcharts.com/sp500/performance"]

    def parse(self, response):
        rows = response.xpath('//table[contains(@class, "table-hover")]//tr[td]')
        for row in rows:
            stock = Sp500ScraperItem()
            stock['rank'] = row.xpath('./td[1]/text()').get().strip()
            stock['company'] = row.xpath('./td[2]/a/text()').get().strip()
            stock['symbol'] = row.xpath('./td[3]/a/text()').get().strip()
            stock['ytd_return'] = ''.join(row.xpath('./td[4]//text()').getall()).strip()
            yield stock


