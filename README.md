# 🏦 YuFinance AI Agent

> 一个基于 **Hand-written ReAct 引擎** 与 **高级 RAG 管道** 的垂直领域金融理财智能助手。

本项目旨在解决传统金融应用中信息零散、私有知识难查询、复杂测算易出错的痛点。通过“Agent（行动力）+ RAG（专业度）”双引擎架构，实现从实时看盘、专业问答到精准测算的闭环金融服务。

---

## 🚀 项目核心亮点

### 1. 核心自研：白盒化 ReAct 执行引擎
不同于市面上的黑盒 Agent 框架，本项目从零手写了三层继承架构的 Agent 执行引擎：
- **BaseAgent**: 维护状态机（IDLE/RUNNING/FINISHED）与核心执行循环。
- **ReActAgent**: 实现标准的 `Think -> Act -> Observe` 推理逻辑。
- **ToolCallAgent**: 深度对接模型原生 Function Calling，支持动态工具分合。
- **思考可视化**: 基于 Python Generator 实现 SSE（Server-Sent Events）流式推流，前端可实时观测智能体的“内心独白”。

### 2. 高级 RAG：金融知识精准召回
针对金融领域长尾词多、专业性强的特点，构建了工业级的 RAG 管道：
- **混合检索 (Hybrid Search)**: 向量检索 + BM25 关键词检索，通过 RRF (Reciprocal Rank Fusion) 算法融合调优。
- **纠正式 RAG (CRAG)**: 引入检索评估逻辑，当内部知识库无法提供高质量支持时，自动触发互联网搜索降级补位。
- **查询改写 (Query Rewriter)**: 自动将用户模糊的输入重写为更具检索亲和力的专业术语。
- **量化评估 (RAGAS)**: 集成 RAGAS 框架，对上下文精度（Precision）和忠实度（Faithfulness）进行量化测评。

### 3. 金融级合规与精准测算
- **计算能力剥离**: 识别房贷、收益率等计算需求后，将参数剥离给 Python 原生财务引擎执行，彻底杜绝大模型的“数学幻觉”。
- **合规隔离墙**: 结合 System Prompt 强制约束与底层工具白名单，严控“买卖建议”风险，确保系统运行在安全围栏内。
- **多用户会话持久化**: 通过 `session_id` 物理隔离不同用户的 JSON 对话历史，支持跨进程会话恢复。

---

## 🛠️ 技术栈

- **大模型核心**: DashScope (通义千问 Qwen-2.5), LangChain
- **向量数据库**: ChromaDB
- **后端框架**: FastAPI (Async, SSE)
- **金融数据**: AKShare (A 股/基金实时数据)
- **工具库**: ReportLab (PDF 生成), Selenium/BeautifulSoup (网页爬取), Baidu Search API

---

## 📂 目录结构

```text
financeHelper/
├── agent/              # 自研 Agent 核心路由逻辑 (Base/ReAct/ToolCall)
├── rag/                # 高级 RAG 流程控制 (CRAG/Rerank/QueryRewrite)
├── tools/              # 金融专用工具集 (股票/新闻/计算器/PDF)
├── app/                # 业务逻辑集成层 (FinanceApp)
├── documents/          # 私有理财产品/内部政策知识库
├── api.py              # FastAPI 接口定义与 SSE 转发逻辑
├── main.py             # 交互式测试入口
└── requirements.txt    # 项目依赖
```

---

## ⚙️ 快速开始

### 1. 环境准备
```bash
git clone <repository-url>
cd financeHelper
pip install -r requirements.txt
```

### 2. 配置秘钥
在项目根目录创建 `.env` 文件，补充以下配置：
```env
DASHSCOPE_API_KEY=your_key_here
BAIDU_SEARCH_API_KEY=your_key_here
BAIDU_SEARCH_BASE_URL=https://api.baidu.com/...
```

### 3. 运行项目
- **启动 API 服务**: `python api.py`
- **启动交互式 Demo**: `python main.py`

---

## 📊 面试核心 Q&A 预演

- **Q: 为什么不用 LangGraph？**
  - A: 为了实现更高的掌控力。手写三层架构可以精准控制 Token 消耗、错误拦截，并完美适配自定义的 SSE 流式思考过程展示。
- **Q: 混合检索的效果提升了多少？**
  - A: 在针对《银行内部贷款准则》这种含大量特定专有名词的文档时，BM25 弥补了向量搜索在关键词匹配上的不足，多路召回显著提升了召回率（Recall）。
- **Q: 如何处理模型幻觉？**
  - A: 采用 CRAG 逻辑做检索后评估，并对高精度计算需求执行“逻辑下沉”，交由 Python 确定性代码处理。

---
**Disclaimer**: 本项目所有金融理财数据及建议仅供演示参考，不构成任何投资指导。

