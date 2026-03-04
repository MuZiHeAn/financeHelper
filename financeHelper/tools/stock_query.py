"""
股票行情查询工具 - 金融 Agent 专用

使用 akshare 库获取 A 股实时/历史行情数据（免费，无需 API Key）

面试话术:
  "我用 akshare 这个开源数据库做股票行情查询，它封装了沪深交易所的公开数据接口，
   无需申请 API Key，覆盖了 A 股全部股票的实时价格、K线、基本面数据。"

安装: pip install akshare
"""
import logging

logger = logging.getLogger(__name__)


class StockQueryTool:
    """A股股票行情查询工具（基于 akshare）"""

    def get_stock_quote(self, symbol: str) -> str:
        """
        查询股票实时行情

        Args:
            symbol: 股票代码，如 '600036'（招商银行）、'000001'（平安银行）
                    也可以是股票名称，如 '招商银行'

        Returns:
            股票行情字符串（价格、涨跌幅、成交量等）
        """
        try:
            import akshare as ak

            # akshare 需要带市场前缀：sh=上海，sz=深圳
            # 先尝试直接用代码查，再尝试名字搜索
            code = symbol.strip()

            # 如果是纯数字代码，加前缀
            if code.isdigit():
                if code.startswith("6"):
                    full_code = f"sh{code}"
                else:
                    full_code = f"sz{code}"

                # 获取实时行情
                df = ak.stock_zh_a_spot_em()
                # 根据代码过滤
                row = df[df["代码"] == code]
                if row.empty:
                    return f"未找到股票代码 {code}，请确认代码是否正确。"
            else:
                # 按名称搜索
                df = ak.stock_zh_a_spot_em()
                row = df[df["名称"].str.contains(code, na=False)]
                if row.empty:
                    return f"未找到名称含 '{code}' 的股票，请尝试使用股票代码。"
                row = row.head(1)  # 取第一个匹配

            r = row.iloc[0]
            result = (
                f"📈 股票行情查询结果\n"
                f"股票：{r.get('名称', 'N/A')}（{r.get('代码', 'N/A')}）\n"
                f"最新价：{r.get('最新价', 'N/A')} 元\n"
                f"涨跌额：{r.get('涨跌额', 'N/A')} 元  |  涨跌幅：{r.get('涨跌幅', 'N/A')}%\n"
                f"今开：{r.get('今开', 'N/A')} 元  |  昨收：{r.get('昨收', 'N/A')} 元\n"
                f"最高：{r.get('最高', 'N/A')} 元  |  最低：{r.get('最低', 'N/A')} 元\n"
                f"成交量：{r.get('成交量', 'N/A')} 手  |  成交额：{r.get('成交额', 'N/A')} 元\n"
                f"市盈率(动)：{r.get('市盈率-动态', 'N/A')}  |  市净率：{r.get('市净率', 'N/A')}\n"
                f"\n⚠️ 免责声明：以上数据仅供参考，不构成投资建议。投资有风险，入市需谨慎。"
            )
            return result

        except ImportError:
            return "错误：akshare 未安装，请运行 `pip install akshare` 后重试。"
        except Exception as e:
            logger.error(f"查询股票行情失败: {e}")
            return f"查询股票行情时发生错误：{str(e)}。请确认股票代码格式（如：600036 或 招商银行）。"

    def get_stock_quote_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "getStockQuote",
                "description": (
                    "查询 A 股股票的实时行情数据，包括价格、涨跌幅、成交量、市盈率等。"
                    "支持股票代码（如 600036）或股票名称（如 招商银行）。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "股票代码（如 '600036'）或股票名称（如 '招商银行'）"
                        }
                    },
                    "required": ["symbol"]
                }
            }
        }
