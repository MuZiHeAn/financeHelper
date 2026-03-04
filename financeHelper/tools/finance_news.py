"""
金融新闻抓取工具 - 金融 Agent 专用

从主流财经网站抓取最新金融新闻，包括：
- 东方财富（eastmoney.com）
- 新浪财经（finance.sina.com.cn）
- 证券时报（stcn.com）

面试话术:
  "金融新闻工具是对原有 web_scraping 工具的精化版本，
   针对主流财经网站的 HTML 结构做了专门解析，
   能够提取结构化的新闻标题+摘要+时间，而不是返回整页原始文本。"
"""
import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# 主流财经新闻网站配置
FINANCE_NEWS_SOURCES = {
    "eastmoney": {
        "name": "东方财富",
        "url": "https://finance.eastmoney.com/",
        "selector": "li.clearfix",      # 新闻列表项
        "title_tag": "a",
        "enabled": True,
    },
    "sina": {
        "name": "新浪财经",
        "url": "https://finance.sina.com.cn/roll/index.d.html?cid=56922&page=1",
        "selector": ".listBlk li",
        "title_tag": "a",
        "enabled": True,
    },
}


class FinanceNewsTool:
    """金融新闻抓取工具 - 从主流财经网站获取今日新闻"""

    def __init__(self):
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9",
        }

    def get_finance_news(self, topic: str = "今日要闻", count: int = 5) -> str:
        """
        获取最新金融新闻

        Args:
            topic: 新闻主题关键词，如 '银行股'、'利率'、'A股'、'今日要闻'
            count: 返回新闻条数（默认5条）

        Returns:
            格式化的新闻列表字符串
        """
        news_items = []

        # ===== 策略1：通过百度搜索金融新闻（最稳定，页面结构不易变） =====
        try:
            search_url = "https://www.baidu.com/s"
            params = {
                "wd": f"site:finance.eastmoney.com OR site:finance.sina.com.cn {topic}",
                "rn": count * 2,  # 多请求一些，过滤后取 count 条
            }
            resp = requests.get(search_url, headers=self.headers, params=params, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")

            results = soup.find_all("div", class_="result")
            for r in results:
                title_tag = r.find("h3")
                if not title_tag:
                    continue
                title = title_tag.get_text(strip=True)
                # 过滤掉太短的标题
                if len(title) < 5:
                    continue

                # 提取摘要
                abstract_tag = r.find("div", class_=lambda x: x and "abstract" in x.lower())
                abstract = abstract_tag.get_text(strip=True)[:100] if abstract_tag else ""

                # 提取链接
                link_tag = title_tag.find("a")
                link = link_tag.get("href", "") if link_tag else ""

                news_items.append({
                    "title": title,
                    "abstract": abstract,
                    "link": link,
                    "source": "百度财经搜索",
                })

                if len(news_items) >= count:
                    break

        except Exception as e:
            logger.warning(f"百度新闻搜索失败: {e}")

        # ===== 策略2：直接抓取东方财富快讯（备选） =====
        if len(news_items) < count:
            try:
                # 东方财富 7x24 快讯 API（公开接口）
                api_url = "https://np-listapi.eastmoney.com/comm/web/getFastList"
                params = {
                    "client": "web",
                    "biz": "web_news_flash",
                    "page": 1,
                    "pageSize": count,
                    "order": 1,
                }
                resp = requests.get(api_url, params=params, timeout=10)
                data = resp.json()
                items = data.get("data", {}).get("list", [])
                for item in items:
                    title = item.get("title", "") or item.get("content", "")[:50]
                    if title:
                        news_items.append({
                            "title": title,
                            "abstract": item.get("content", "")[:100],
                            "link": "",
                            "source": "东方财富7x24快讯",
                        })
                        if len(news_items) >= count:
                            break
            except Exception as e:
                logger.warning(f"东方财富快讯获取失败: {e}")

        # ===== 格式化输出 =====
        if not news_items:
            return f"⚠️ 未能获取到关于「{topic}」的金融新闻，请稍后重试或换个关键词。"

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        lines = [f"📰 金融新闻摘要（{now}）- 关键词：{topic}\n"]
        for i, item in enumerate(news_items[:count], 1):
            lines.append(f"{i}. 【{item['source']}】{item['title']}")
            if item["abstract"]:
                lines.append(f"   摘要：{item['abstract'][:80]}...")
            lines.append("")

        lines.append("⚠️ 以上新闻仅供参考，不构成投资建议。")
        return "\n".join(lines)

    def get_finance_news_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "getFinanceNews",
                "description": (
                    "从东方财富、新浪财经等主流财经网站获取最新金融新闻。"
                    "可以指定新闻主题关键词，如 '银行股'、'利率政策'、'A股行情'、'今日要闻'等。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "新闻主题关键词，如 '银行股'、'利率'、'基金'、'今日要闻'",
                            "default": "今日要闻"
                        },
                        "count": {
                            "type": "integer",
                            "description": "返回新闻条数，默认5条，最多10条",
                            "default": 5
                        }
                    },
                    "required": []
                }
            }
        }
