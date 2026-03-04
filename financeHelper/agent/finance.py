"""
YuFinance 金融智能体 - 在 YuManus 基础上改造

改造说明（最小改动原则）：
  - BaseAgent / ReActAgent / ToolCallAgent 一行不动
  - 只替换：系统提示词 + 工具集
  - 新增 3 个金融工具：股票行情、金融新闻、财务计算
  - 移除：终端命令（合规风险）、资源下载（金融场景用处不大）
  - 保留：文件读写、网页爬取、百度搜索、PDF生成、终止

面试话术:
  "这套金融 Agent 是在原有 ReAct 框架上做了最小化改动——
   底层的状态机、执行循环、Function Calling 机制完全不变，
   只换了业务层的工具集和系统提示词。
   这体现了四层继承设计的好处：新业务场景只需要在最顶层做替换，
   不影响底层稳定的核心逻辑。"
"""
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tool_call_agent import ToolCallAgent
from tools.file_operation import FileOperationTool
from tools.web_search import BaiduSearchTool
from tools.web_scraping import WebScrapingTool
from tools.pdf_generation import PDFGenerationTool
from tools.terminate import TerminateTool
from tools.resource_download import ResourceDownloadTool   # 下载年报/基金公告/研报 PDF
# ↓ 新增的金融专用工具
from tools.stock_query import StockQueryTool
from tools.finance_news import FinanceNewsTool
from tools.finance_calculator import FinanceCalculatorTool
from tools.internal_knowledge import InternalKnowledgeTool
from rag.document_loader import load_markdown_documents
from rag.vector_store import create_vector_store
import config

logger = logging.getLogger(__name__)


# ===== 金融版系统提示词 =====
FINANCE_SYSTEM_PROMPT = """你是 Finance，一个专业的金融信息助手，专注于以下领域：

1. 📈 A股行情查询：提供股票实时价格、涨跌幅、基本面数据（PE、PB等）
2. 📰 金融资讯：从东方财富、新浪财经等主流财经媒体获取最新市场动态
3. 🏦 银行业务咨询：储蓄产品、贷款利率、理财规划等问题解答
4. 🧮 财务计算：房贷月供、存款收益、年化收益率、复利等精确计算
5. 🔍 金融知识：解释金融概念、分析市场现象、科普投资基础知识

【重要合规约束 - 请严格遵守】
- 所有股票信息、行情数据仅供参考，不构成投资建议
- 不对任何股票做明确的买入/卖出推荐
- 涉及投资建议时，必须附加免责声明："投资有风险，入市需谨慎"
- 对于明显超出金融范畴的请求，礼貌说明并引导回到金融主题

你只有 20 次工具调用机会，请高效完成用户需求。"""

# ===== 金融版下一步引导提示词 =====
FINANCE_NEXT_STEP_PROMPT = """根据用户的金融相关需求，优先使用以下策略：
- 查股票行情 → 使用 getStockQuote 工具
- 查金融新闻 → 使用 getFinanceNews 工具  
- 计算贷款/存款/收益 → 使用 calculateFinance 工具
- 咨询内部政策/产品详情 → 使用 queryInternalKnowledge 工具
- 深入了解某只股票 → 先用 getStockQuote 拿行情，再用 scrapeWebPage 看详情
- 生成分析报告 → 用 generatePDF 保存结果

完成任务后调用 doTerminate 结束对话。如有不确定的金融信息，请注明来源并提醒用户自行核实。"""


def create_yu_finance() -> ToolCallAgent:
    """
    创建 Finance 金融智能体

    改造对比（和原 create_yu_manus 的差异）：
      移除: ResourceDownloadTool（金融场景无用）
      移除: TerminalOperationTool（合规风险，金融系统不允许随意执行命令）
      新增: StockQueryTool（A股行情）
      新增: FinanceNewsTool（金融新闻）
      新增: FinanceCalculatorTool（财务计算）
      新增: InternalKnowledgeTool（私有 RAG 知识库查询）
      修改: system_prompt → 金融专属角色定义和合规约束
    """
    # ── 1. 实例化工具 ──────────────────────────────────────────────
    file_tool = FileOperationTool()
    search_tool = BaiduSearchTool(config.BAIDU_SEARCH_API_KEY, config.BAIDU_SEARCH_BASE_URL)
    scraping_tool = WebScrapingTool()
    pdf_tool = PDFGenerationTool()
    terminate_tool = TerminateTool()
    download_tool = ResourceDownloadTool()   # 可下载年报、基金招募说明书、研究报告等 PDF
    # 金融新工具
    stock_tool = StockQueryTool()
    news_tool = FinanceNewsTool()
    calc_tool = FinanceCalculatorTool()
    
    # 初始化 RAG 向量库用于工具调用
    documents = load_markdown_documents()
    vector_store = create_vector_store(documents) if documents else create_vector_store()
    knowledge_tool = InternalKnowledgeTool(vector_store)

    # ── 2. 工具 JSON Schema 列表（给模型看） ─────────────────────────
    tool_definitions = [
        # 金融核心工具（放前面，模型优先看到）
        stock_tool.get_stock_quote_schema(),
        news_tool.get_finance_news_schema(),
        calc_tool.calculate_finance_schema(),
        knowledge_tool.get_schema(),
        # 通用辅助工具
        search_tool.search_schema(),
        scraping_tool.scrape_schema(),
        download_tool.download_schema(),      # 下载年报/研报 PDF
        file_tool.read_file_schema(),
        file_tool.write_file_schema(),
        file_tool.append_to_file_schema(),
        pdf_tool.generate_schema(),
        terminate_tool.terminate_schema(),
    ]

    # ── 3. 创建 Agent（完全复用 ToolCallAgent，不改任何底层逻辑）────
    agent = ToolCallAgent(
        tools=tool_definitions,
        api_key=config.DASHSCOPE_API_KEY,
        model_name=config.CHAT_MODEL_NAME,
    )

    # ── 4. 注册工具执行函数 ───────────────────────────────────────────
    # 金融新工具
    agent.register_tool_function("getStockQuote", stock_tool.get_stock_quote)
    agent.register_tool_function("getFinanceNews", news_tool.get_finance_news)
    agent.register_tool_function("calculateFinance", calc_tool.calculate_finance)
    agent.register_tool_function("queryInternalKnowledge", knowledge_tool.query_knowledge)
    # 通用工具
    agent.register_tool_function("search", search_tool.search)
    agent.register_tool_function("scrapeWebPage", scraping_tool.scrape_web_page)
    agent.register_tool_function("downloadResource", download_tool.download_resource)  # 下载年报/研报
    agent.register_tool_function("readFile", file_tool.read_file)
    agent.register_tool_function("writeFile", file_tool.write_file)
    agent.register_tool_function("appendToFile", file_tool.append_to_file)
    agent.register_tool_function("generatePDF", pdf_tool.generate_pdf)
    agent.register_tool_function("doTerminate", terminate_tool.do_terminate)

    # ── 5. 配置金融版提示词和名称 ────────────────────────────────────
    # 核心逻辑等于：messages = [
    # {"role": "system", "content": FINANCE_SYSTEM_PROMPT},
    # {"role": "system", "content": FINANCE_NEXT_STEP_PROMPT},
    # {"role": "user", "content": "查一下招商银行股价"}

    agent.name = "Finance"
    agent.system_prompt = FINANCE_SYSTEM_PROMPT
    agent.next_step_prompt = FINANCE_NEXT_STEP_PROMPT
    agent.max_steps = config.AGENT_MAX_STEPS

    logger.info("Finance 金融智能体创建完成，已注册 11 个工具（3个金融专用 + 8个通用）")
    return agent
