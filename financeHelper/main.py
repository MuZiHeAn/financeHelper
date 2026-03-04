"""
主入口 - 交互式测试菜单

提供六种模式：
1. YuManus Agent 对话（多工具调用）
2. LoveApp 基础对话（多轮记忆）
3. LoveApp 基础 RAG 知识库问答
4. ⭐ 高级 RAG 对话（Hybrid Search + Re-Ranking + CRAG）
5. 📊 RAG 评估测试（RAGAS 指标）
6. 💹 YuFinance 金融智能体（股票行情 + 金融新闻 + 财务计算）

运行方式: python main.py
"""
import sys
import os
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def test_agent():
    """测试 YuManus Agent（多工具调用）"""
    from agent.yu_manus import create_yu_manus

    print("\n" + "=" * 60)
    print("🤖 YuManus 超级智能体 (输入 'quit' 退出)")
    print("=" * 60)

    agent = create_yu_manus()

    while True:
        user_input = input("\n👤 你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        print("\n🤖 YuManus 正在处理...\n")
        for chunk in agent.stream_run(user_input):
            print(chunk, end="", flush=True)
        print()
        agent = create_yu_manus()


def test_finance_app_chat():
    """测试 FinanceApp 基础对话"""
    from app.finance_app import FinanceApp

    print("\n" + "=" * 60)
    print("🏦 金融理财专家对话 (输入 'quit' 退出)")
    print("=" * 60)

    app = FinanceApp()
    chat_id = "test_user_001"

    while True:
        user_input = input("\n👤 你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        print("\n🏦 理财专家: ", end="", flush=True)
        result = app.do_chat(user_input, chat_id)
        print(result)


def test_finance_app_rag():
    """测试 FinanceApp 基础 RAG 知识库问答"""
    from app.finance_app import FinanceApp

    print("\n" + "=" * 60)
    print("📚 金融知识库问答 - 基础版 (输入 'quit' 退出)")
    print("=" * 60)

    app = FinanceApp()
    chat_id = "rag_user_001"

    status_filter = _select_status()

    while True:
        user_input = input("\n👤 你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        print("\n📚 知识库回答: ", end="", flush=True)
        result = app.do_chat_with_rag(user_input, chat_id, status_filter)
        print(result)


def test_advanced_rag():
    """
    ⭐ 测试高级 RAG（Hybrid Search + Re-Ranking + CRAG）

    这是整个项目的核心亮点！
    完整管道: 混合检索 → 重排序 → 检索评估 → 纠正式回答
    """
    from rag.document_loader import load_markdown_documents
    from rag.vector_store import create_vector_store
    from rag.crag import CRAGPipeline

    print("\n" + "=" * 60)
    print("⭐ 高级 RAG 对话 - CRAG 纠正式检索增强 (输入 'quit' 退出)")
    print("=" * 60)
    print()
    print("📌 管道流程:")
    print("  用户查询 → 混合检索(BM25+向量+RRF)")
    print("         → 重排序(LLM Reranker)")
    print("         → 检索评估(质量判断)")
    print("         → CORRECT: 直接用知识库回答")
    print("         → AMBIGUOUS: 知识库+Web搜索补充")
    print("         → INCORRECT: 降级到Web搜索")
    print()

    # 初始化
    print("⏳ 正在加载知识库和构建索引...")
    documents = load_markdown_documents()
    vector_store = create_vector_store(documents)
    pipeline = CRAGPipeline(documents, vector_store)
    print("✅ 初始化完成！\n")

    status_filter = _select_status()

    while True:
        user_input = input("\n👤 你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        print("\n⭐ CRAG 正在处理（请观察日志中的完整管道流程）...\n")
        result = pipeline.ask(user_input, status_filter)
        print(f"\n💬 最终回答:\n{result}")


def test_ragas_evaluation():
    """
    📊 测试 RAGAS 评估

    自动运行一组测试问题，评估基础 RAG vs 高级 RAG 的质量差异
    """
    from rag.document_loader import load_markdown_documents
    from rag.vector_store import create_vector_store
    from rag.crag import CRAGPipeline
    from rag.rag_chain import ask_with_rag
    from rag.ragas_eval import RAGEvaluator

    print("\n" + "=" * 60)
    print("📊 RAGAS 评估 - 基础 RAG vs 高级 RAG 对比")
    print("=" * 60)

    # 预定义测试问题
    test_questions = [
        "降息环境下如何配置资产？",
        "房贷提前还款划算吗？",
        "理财产品打破刚兑意味着什么？",
    ]

    print("\n⏳ 正在初始化...")
    documents = load_markdown_documents()
    vector_store = create_vector_store(documents)
    crag_pipeline = CRAGPipeline(documents, vector_store)
    evaluator = RAGEvaluator()

    print(f"✅ 初始化完成，将对 {len(test_questions)} 个测试问题进行评估\n")

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"🔬 测试问题 {i}/{len(test_questions)}: {question}")
        print(f"{'='*60}")

        # 基础 RAG 回答
        print("\n📝 基础 RAG 正在回答...")
        basic_answer = ask_with_rag(vector_store, question)
        print(f"  基础回答: {basic_answer[:100]}...")

        # 高级 RAG (CRAG) 回答
        print("\n⭐ 高级 RAG (CRAG) 正在回答...")
        advanced_answer = crag_pipeline.ask(question)
        print(f"  高级回答: {advanced_answer[:100]}...")

        # 获取检索上下文用于评估
        from rag.vector_store import get_retriever
        retriever = get_retriever(vector_store, top_k=3)
        docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in docs]

        # 评估基础 RAG
        print("\n📊 评估基础 RAG:")
        basic_eval = evaluator.evaluate(question, basic_answer, contexts)
        print(basic_eval.summary())

        # 评估高级 RAG
        print("\n📊 评估高级 RAG (CRAG):")
        advanced_eval = evaluator.evaluate(question, advanced_answer, contexts)
        print(advanced_eval.summary())

        # 对比
        diff = advanced_eval.overall_score() - basic_eval.overall_score()
        emoji = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
        print(f"\n{emoji} 综合得分变化: {diff:+.2f} "
              f"(基础: {basic_eval.overall_score():.2f} → "
              f"高级: {advanced_eval.overall_score():.2f})")


def test_finance_agent():
    """
    💹 测试 YuFinance 金融智能体

    改造自 YuManus，底层完全不变，只换了工具集和系统提示词：
    新工具: 股票行情(akshare) + 金融新闻 + 财务计算
    测试问题示例:
      - 查一下招商银行的股票行情
      - 今天银行股有什么新闻？
      - 100万贷款30年月供是多少？
      - 帮我算一下10万存一年定期利息
    """
    from agent.yu_finance import create_yu_finance

    print("\n" + "=" * 60)
    print("💹 YuFinance 金融智能体 (输入 'quit' 退出)")
    print("=" * 60)
    print("\n💡 示例问题:")
    print("  • 查一下招商银行的股票行情")
    print("  • 今天有什么银行股相关新闻？")
    print("  • 贷款100万，30年，利率3.95%，月供多少？")
    print("  • 10万元存一年定期，利率1.8%，到期收益多少？")
    print("  • 帮我分析一下平安银行最近的情况\n")

    agent = create_yu_finance()

    while True:
        user_input = input("\n👤 你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        print("\n💹 YuFinance 正在处理...\n")
        for chunk in agent.stream_run(user_input):
            print(chunk, end="", flush=True)
        print()
        agent = create_yu_finance()  # 每轮重置，避免历史干扰



def _select_status() -> str:
    """选择咨询类型（状态过滤）"""
    print("\n请选择咨询类型:")
    print("  1. 储蓄")
    print("  2. 贷款")
    print("  3. 投资")
    status_map = {"1": "储蓄", "2": "贷款", "3": "投资"}
    choice = input("请选择 (1/2/3, 默认1): ").strip()
    status_filter = status_map.get(choice, "储蓄")
    print(f"已选择: {status_filter}")
    return status_filter



def main():
    """主菜单"""
    print("=" * 60)
    print("  🚀 YuAiAgent Python 版 - AI 核心模块")
    print("=" * 60)
    print()
    print("  对应 Java 项目: yu-ai-agent (Spring AI)")
    print("  Python 框架: LangChain + DashScope")
    print()
    print("  📌 高级 RAG 技术栈:")
    print("     • Hybrid Search (BM25 + Vector + RRF)")
    print("     • Re-Ranking (LLM Cross-Encoder)")
    print("     • CRAG (Corrective RAG, 论文: arXiv:2401.15884)")
    print("     • RAGAS 评估体系")
    print()

    while True:
        print("\n" + "-" * 40)
        print("请选择测试模式:")
        print("  1. 🤖 YuManus Agent (多工具调用)")
        print("  2. 💕 LoveApp 基础对话 (多轮记忆)")
        print("  3. 📚 LoveApp 基础 RAG 知识库问答")
        print("  4. ⭐ 高级 RAG 对话 (Hybrid+Rerank+CRAG)")
        print("  5. 📊 RAG 评估对比 (RAGAS 指标)")
        print("  6. 💹 YuFinance 金融智能体 (股票/新闻/计算)")
        print("  0. 退出")
        print("-" * 40)

        choice = input("请输入选项 (0-6): ").strip()

        if choice == "1":
            test_agent()
        elif choice == "2":
            test_love_app_chat()
        elif choice == "3":
            test_love_app_rag()
        elif choice == "4":
            test_advanced_rag()
        elif choice == "5":
            test_ragas_evaluation()
        elif choice == "6":
            test_finance_agent()
        elif choice == "0":
            print("\n再见！👋")
            break
        else:
            print("无效选项，请重新输入")


if __name__ == "__main__":
    main()
