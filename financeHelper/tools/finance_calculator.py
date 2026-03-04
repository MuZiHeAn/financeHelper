"""
金融计算工具 - 金融 Agent 专用

提供常用银行/金融计算功能：
  - 房贷月供（等额还款）
  - 存款到期收益（定期/活期）
  - 年化收益率计算
  - 复利终值计算
  - 简单的股票市盈率估值

面试话术:
  "银行业务场景里，用户经常问'我存10万定期一年能拿多少利息'、
   '100万贷款30年月供多少'这类问题。我给 Agent 加了金融计算工具，
   让它能直接计算精确数字，不靠大模型'猜'答案，
   这样就消除了数值计算上的幻觉问题。"
"""
import math
import logging
import json

logger = logging.getLogger(__name__)


class FinanceCalculatorTool:
    """金融计算工具 - 精确计算银行/投资相关数值"""

    def calculate_finance(self, calc_type: str, params: str) -> str:
        """
        金融计算入口，根据类型调用不同的计算函数

        Args:
            calc_type: 计算类型
                - "mortgage"   : 房贷月供（等额本息）
                - "deposit"    : 存款利息
                - "annualized" : 年化收益率
                - "compound"   : 复利终值
                - "pe_val"     : 市盈率估值
            params: JSON字符串，包含各类型所需的参数

        Returns:
            计算结果字符串
        """
        try:
            p = json.loads(params)
        except (json.JSONDecodeError, TypeError):
            return f"参数格式错误：params 需要是 JSON 格式字符串，如 '{{\"amount\": 1000000}}'"

        calc_type = calc_type.lower().strip()

        if calc_type == "mortgage":
            return self._mortgage(p)
        elif calc_type == "deposit":
            return self._deposit(p)
        elif calc_type == "annualized":
            return self._annualized(p)
        elif calc_type == "compound":
            return self._compound(p)
        elif calc_type == "pe_val":
            return self._pe_valuation(p)
        else:
            return (
                f"不支持的计算类型：'{calc_type}'。\n"
                "支持的类型：mortgage（房贷月供）、deposit（存款利息）、"
                "annualized（年化收益率）、compound（复利终值）、pe_val（市盈率估值）"
            )

    # ---------- 各计算函数 ----------

    def _mortgage(self, p: dict) -> str:
        """
        等额本息房贷月供计算

        params 示例: {"amount": 1000000, "annual_rate": 3.95, "years": 30}
        """
        try:
            amount = float(p["amount"])          # 贷款总额（元）
            annual_rate = float(p["annual_rate"])  # 年利率（%）
            years = int(p["years"])              # 还款年数
        except (KeyError, ValueError):
            return "参数缺失：mortgage 需要 amount（贷款额）、annual_rate（年利率%）、years（年数）"

        monthly_rate = annual_rate / 100 / 12   # 月利率
        n = years * 12                           # 还款总月数

        if monthly_rate == 0:
            monthly_payment = amount / n
        else:
            # 等额本息公式: M = P * r * (1+r)^n / ((1+r)^n - 1)
            monthly_payment = amount * monthly_rate * (1 + monthly_rate) ** n / ((1 + monthly_rate) ** n - 1)

        total_payment = monthly_payment * n
        total_interest = total_payment - amount

        return (
            f"🏠 房贷月供计算结果\n"
            f"贷款金额：{amount:,.0f} 元\n"
            f"年利率：{annual_rate}%  |  还款期限：{years} 年（{n} 个月）\n"
            f"─────────────────────────\n"
            f"每月还款：{monthly_payment:,.2f} 元\n"
            f"还款总额：{total_payment:,.2f} 元\n"
            f"支付利息：{total_interest:,.2f} 元\n"
            f"利息占比：{total_interest/amount*100:.1f}%\n"
        )

    def _deposit(self, p: dict) -> str:
        """
        存款利息计算（定期/活期）

        params 示例: {"amount": 100000, "annual_rate": 2.0, "months": 12}
        """
        try:
            amount = float(p["amount"])
            annual_rate = float(p["annual_rate"])
            months = int(p.get("months", 12))
        except (KeyError, ValueError):
            return "参数缺失：deposit 需要 amount（本金）、annual_rate（年利率%）、months（存款月数，默认12）"

        # 定期存款：单利
        interest = amount * (annual_rate / 100) * (months / 12)
        total = amount + interest

        return (
            f"💰 存款收益计算结果\n"
            f"存款本金：{amount:,.0f} 元\n"
            f"年利率：{annual_rate}%（参考利率，实际以银行公告为准）\n"
            f"存款期限：{months} 个月\n"
            f"─────────────────────────\n"
            f"利息收入：{interest:,.2f} 元\n"
            f"税后利息：{interest * 0.8:,.2f} 元（扣除20%利息税，部分银行代扣）\n"
            f"到期总额：{total:,.2f} 元\n"
            f"\n当前参考利率：一年定期约 1.5%-2.0%（实际利率请以各银行最新公告为准）"
        )

    def _annualized(self, p: dict) -> str:
        """
        年化收益率计算

        params 示例: {"cost": 10000, "current": 11500, "days": 365}
        """
        try:
            cost = float(p["cost"])
            current = float(p["current"])
            days = int(p.get("days", 365))
        except (KeyError, ValueError):
            return "参数缺失：annualized 需要 cost（成本）、current（现值）、days（持有天数，默认365）"

        total_return = (current - cost) / cost * 100
        annualized = total_return / days * 365

        return (
            f"📊 年化收益率计算结果\n"
            f"买入成本：{cost:,.2f} 元\n"
            f"当前价值：{current:,.2f} 元\n"
            f"持有天数：{days} 天\n"
            f"─────────────────────────\n"
            f"总收益率：{total_return:.2f}%\n"
            f"年化收益率：{annualized:.2f}%\n"
        )

    def _compound(self, p: dict) -> str:
        """
        复利终值计算

        params 示例: {"principal": 10000, "annual_rate": 8.0, "years": 10}
        """
        try:
            principal = float(p["principal"])
            annual_rate = float(p["annual_rate"])
            years = int(p["years"])
        except (KeyError, ValueError):
            return "参数缺失：compound 需要 principal（本金）、annual_rate（年利率%）、years（年数）"

        # 复利公式: FV = PV * (1 + r)^n
        fv = principal * (1 + annual_rate / 100) ** years
        total_interest = fv - principal

        return (
            f"📈 复利计算结果（72法则：资产翻倍需要 {72/annual_rate:.0f} 年）\n"
            f"初始本金：{principal:,.0f} 元\n"
            f"年化收益率：{annual_rate}%\n"
            f"投资年限：{years} 年\n"
            f"─────────────────────────\n"
            f"到期终值：{fv:,.2f} 元\n"
            f"收益总额：{total_interest:,.2f} 元\n"
            f"资产增长：{fv/principal:.2f} 倍\n"
        )

    def _pe_valuation(self, p: dict) -> str:
        """
        股票市盈率估值参考

        params 示例: {"eps": 2.5, "pe_ratio": 15}
        """
        try:
            eps = float(p["eps"])           # 每股收益（EPS）
            pe_ratio = float(p["pe_ratio"]) # 目标市盈率
        except (KeyError, ValueError):
            return "参数缺失：pe_val 需要 eps（每股收益）、pe_ratio（目标市盈率）"

        target_price = eps * pe_ratio

        pe_ref = {
            "银行股": "5-10倍（低估值行业）",
            "消费股": "20-30倍",
            "科技股": "30-50倍",
            "成长股": "50+倍",
        }

        ref_str = "\n".join(f"  {k}：{v}" for k, v in pe_ref.items())
        return (
            f"🔍 市盈率(PE)估值参考\n"
            f"每股收益(EPS)：{eps} 元\n"
            f"目标市盈率(PE)：{pe_ratio} 倍\n"
            f"─────────────────────────\n"
            f"估算合理股价：{target_price:.2f} 元\n\n"
            f"行业参考市盈率（仅供参考）：\n{ref_str}\n\n"
            f"⚠️ PE 估值法仅适用于盈利稳定的成熟企业，不构成投资建议。"
        )

    def calculate_finance_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "calculateFinance",
                "description": (
                    "金融/银行业务计算工具，支持以下计算类型：\n"
                    "- mortgage: 房贷月供（等额本息）\n"
                    "- deposit: 存款到期收益\n"
                    "- annualized: 投资年化收益率\n"
                    "- compound: 复利终值（长期收益预测）\n"
                    "- pe_val: 股票市盈率估值"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "calc_type": {
                            "type": "string",
                            "description": "计算类型：mortgage / deposit / annualized / compound / pe_val",
                            "enum": ["mortgage", "deposit", "annualized", "compound", "pe_val"]
                        },
                        "params": {
                            "type": "string",
                            "description": (
                                "JSON格式的计算参数。\n"
                                "mortgage示例: '{\"amount\": 1000000, \"annual_rate\": 3.95, \"years\": 30}'\n"
                                "deposit示例:  '{\"amount\": 100000, \"annual_rate\": 2.0, \"months\": 12}'\n"
                                "annualized示例: '{\"cost\": 10000, \"current\": 11500, \"days\": 365}'\n"
                                "compound示例: '{\"principal\": 10000, \"annual_rate\": 8.0, \"years\": 10}'\n"
                                "pe_val示例: '{\"eps\": 2.5, \"pe_ratio\": 15}'"
                            )
                        }
                    },
                    "required": ["calc_type", "params"]
                }
            }
        }
