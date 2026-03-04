"""
百度搜索工具 - 对应 Java BaiduSearchTool.java

通过 SearchAPI 提供的百度搜索接口返回实时搜索结果。
使用 requests 库代替 Java 的 RestTemplate。
"""
import requests
import logging

logger = logging.getLogger(__name__)


class BaiduSearchTool:
    """百度联网搜索工具 - 对应 Java BaiduSearchTool"""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url

    def search(self, query: str) -> str:
        """执行百度搜索 - 对应 Java @Tool search"""
        if not query or not query.strip():
            return "Error: query must not be blank."
        if not self.api_key:
            return "Error: baidu.search.api-key is not configured."

        try:
            params = {
                "engine": "baidu",
                "q": query.strip(),
                "api_key": self.api_key
            }
            response = requests.get(self.base_url, params=params, timeout=30)

            if response.status_code != 200:
                return f"Error: HTTP {response.status_code}"

            body = response.text
            if not body:
                return "No results returned from Baidu."

            # 简单清理：压缩空白字符
            cleaned = " ".join(body.split())
            if len(cleaned) > 600:
                cleaned = cleaned[:600] + "..."
            return cleaned

        except requests.RequestException as e:
            return f"Error calling Baidu search: {e}"

    def search_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search for information from Baidu Search Engine",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query keyword"}
                    },
                    "required": ["query"]
                }
            }
        }
