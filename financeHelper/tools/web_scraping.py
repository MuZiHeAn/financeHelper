"""
网页爬取工具 - 对应 Java WebScrapingTool.java

使用 BeautifulSoup4 代替 Java 的 Jsoup 解析 HTML。
"""
import requests
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


class WebScrapingTool:
    """网页爬取工具 - 对应 Java WebScrapingTool"""

    def scrape_web_page(self, url: str) -> str:
        """爬取网页内容 - 对应 Java @Tool scrapeWebPage"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.encoding = response.apparent_encoding

            soup = BeautifulSoup(response.text, "html.parser")
            # 提取纯文本（去掉 script/style 标签）
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)

            if len(text) > 2000:
                text = text[:2000] + "..."
            return text

        except Exception as e:
            return f"Error scraping web page: {e}"

    def scrape_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "scrapeWebPage",
                "description": "Scrape the content of a web page",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL of the web page to scrape"}
                    },
                    "required": ["url"]
                }
            }
        }
