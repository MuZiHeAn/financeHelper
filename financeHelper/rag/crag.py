"""
===============================================================================
🔄 CRAG（Corrective RAG）纠正式检索增强生成 - 高级 RAG 技术 #3
===============================================================================

📚 论文来源：
   "Corrective Retrieval Augmented Generation"
   Shi-Qi Yan et al., arXiv:2401.15884, January 2024
   GitHub: LangGraph 官方有 CRAG 示例实现

🧠 【核心概念 - 面试最强亮点】

传统 RAG 有一个隐含假设：检索到的文档一定是相关的。
但现实中，检索可能返回不相关的文档（尤其是知识库覆盖不全时），
如果把不相关的文档强行塞给 LLM，反而会导致"幻觉"！

CRAG 的核心创新：在检索和生成之间加一个"检索评估器"（Retrieval Evaluator），
自动判断检索质量，并根据结果采取不同策略：

  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  用户查询 → 检索文档 → 【检索评估器】判断文档质量                    │
  │                           ↓                                     │
  │                    ┌──────┼──────┐                              │
  │                    ↓      ↓      ↓                              │
  │               ✅ CORRECT  ⚠️ AMBIGUOUS  ❌ INCORRECT            │
  │               文档相关    不确定        文档不相关                 │
  │                    ↓      ↓           ↓                         │
  │               直接使用  文档+Web搜索  完全用Web搜索                │
  │                    ↓      ↓           ↓                         │
  │                    └──────┼───────────┘                         │
  │                           ↓                                     │
  │                      提取关键信息                                │
  │                           ↓                                     │
  │                      LLM 生成回答                               │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘

📐 【三个关键组件】

1. Retrieval Evaluator（检索评估器）：
   - 用 LLM 判断每个检索到的文档与查询是否相关
   - 输出三种判断：CORRECT / AMBIGUOUS / INCORRECT
   - 论文中用轻量级模型（如 T5-large）微调

2. Knowledge Refinement（知识精炼）：
   - 对相关文档做"分解-重组"（Decompose-then-Recompose）
   - 把文档拆成小片段 → 过滤不相关片段 → 重组关键信息
   - 这一步去除了"噪声信息"

3. Web Search Fallback（Web搜索降级）：
   - 当内部知识库不足时，自动调用 Web 搜索补充
   - AMBIGUOUS 时：内部文档 + Web 搜索结果一起用
   - INCORRECT 时：完全依赖 Web 搜索

💡 【面试话术】
"我实现了 CRAG（Corrective RAG）论文中的核心思想。
在检索和生成之间加了一个检索评估器，自动评估每个检索文档的相关性。
当评估为 Correct 时直接使用文档；
当评估为 Ambiguous 时补充 Web 搜索；
当评估为 Incorrect 时完全降级到 Web 搜索。
这种自纠正机制有效解决了知识库覆盖不全时的幻觉问题。"

Q: "检索评估器是怎么实现的？"
A: "论文中用微调的 T5 做分类，我的实现中用 LLM 做 few-shot 分类，
   让模型对 (query, document) 对打一个 [1-10] 的相关性分数，
   根据阈值划分为 CORRECT/AMBIGUOUS/INCORRECT 三档。
   生产环境可以替换为微调的小模型来提升速度。"

📖 【学习路径】
1. 先读论文摘要和 Introduction（搞懂动机）
2. 看 LangGraph 官方 CRAG 示例代码（搞懂代码结构）
3. 理解三种判断逻辑和降级策略
4. 论文 arXiv: https://arxiv.org/abs/2401.15884
===============================================================================
"""

import logging
import re
from enum import Enum
from langchain_core.documents import Document
from dashscope import Generation
from rag.hybrid_search import HybridSearchRetriever
from rag.reranker import BatchLLMReranker
from tools.web_search import BaiduSearchTool
import config

logger = logging.getLogger(__name__)


# =============================================================================
# 第一部分：检索质量评估（Retrieval Evaluator）
# =============================================================================

class RetrievalQuality(Enum):
    """
    检索质量枚举 - 对应论文中的三种判断结果

    CORRECT:   文档与查询高度相关，可以直接使用
    AMBIGUOUS: 文档与查询部分相关，需要补充 Web 搜索
    INCORRECT: 文档与查询不相关，需要完全降级到 Web 搜索
    """
    CORRECT = "correct"       # ✅ 相关
    AMBIGUOUS = "ambiguous"   # ⚠️ 不确定
    INCORRECT = "incorrect"   # ❌ 不相关


class RetrievalEvaluator:
    """
    检索评估器 - CRAG 的核心组件

    【工作原理】
    对每个检索到的文档，让 LLM 评估其与查询的相关性。
    返回一个 1-10 的分数，然后根据阈值判断文档质量：

    分数 >= 7 → CORRECT   （文档相关，可以用）
    分数 4-6  → AMBIGUOUS （不确定，需补充）
    分数 <= 3 → INCORRECT （不相关，丢弃）

    【面试要点】
    论文中用微调的 T5-large 做二分类器（相关/不相关），
    我这里用 LLM few-shot 做三分类，效果类似但更灵活。
    实际生产中可以用 BGE-reranker 的分数来代替，速度更快。
    """

    # 分数阈值（可调参数）
    CORRECT_THRESHOLD = 7    # >= 7 分判为 CORRECT
    AMBIGUOUS_THRESHOLD = 4  # >= 4 分判为 AMBIGUOUS, < 4 判为 INCORRECT

    def __init__(self):
        self.api_key = config.DASHSCOPE_API_KEY
        self.model_name = config.CHAT_MODEL_NAME

    def evaluate_document(self, query: str, document: Document) -> tuple[RetrievalQuality, float]:
        """
        评估单个文档的检索质量

        Args:
            query: 用户查询
            document: 待评估的文档
        Returns:
            (RetrievalQuality, score) 质量判断和分数
        """
        content = document.page_content[:400]

        eval_prompt = f"""你是一个文档相关性评估专家。请判断以下文档与查询的相关程度。

【查询】{query}

【文档】{content}

【评分标准】
请给出 1-10 之间的整数评分：
- 8-10: 文档直接回答了查询，信息非常相关
- 5-7: 文档包含部分相关信息，但不完整
- 1-4: 文档与查询几乎不相关

只输出一个数字:"""

        try:
            response = Generation.call(
                api_key=self.api_key,
                model=self.model_name,
                messages=[{"role": "user", "content": eval_prompt}],
                result_format="message",
            )

            if response.status_code == 200:
                text = response.output.choices[0].message.get("content", "5").strip()
                numbers = re.findall(r'\d+', text)
                if numbers:
                    score = min(max(int(numbers[0]), 1), 10)
                else:
                    score = 5

                # 根据阈值判断质量
                if score >= self.CORRECT_THRESHOLD:
                    quality = RetrievalQuality.CORRECT
                elif score >= self.AMBIGUOUS_THRESHOLD:
                    quality = RetrievalQuality.AMBIGUOUS
                else:
                    quality = RetrievalQuality.INCORRECT

                return quality, float(score)

        except Exception as e:
            logger.warning(f"文档评估失败: {e}")

        return RetrievalQuality.AMBIGUOUS, 5.0

    def evaluate_documents(self, query: str, documents: list[Document]) -> tuple[RetrievalQuality, list[Document]]:
        """
        批量评估文档并决定整体检索质量

        【整体判断逻辑（对应论文）】
        - 如果 ≥ 50% 的文档是 CORRECT → 整体 CORRECT
        - 如果所有文档都是 INCORRECT → 整体 INCORRECT
        - 其他情况 → 整体 AMBIGUOUS

        Args:
            query: 用户查询
            documents: 文档列表
        Returns:
            (整体质量判断, 过滤后的相关文档列表)
        """
        if not documents:
            return RetrievalQuality.INCORRECT, []

        correct_docs = []
        ambiguous_docs = []
        incorrect_count = 0

        logger.info(f"📋 评估 {len(documents)} 个检索文档的质量...")

        for i, doc in enumerate(documents):
            quality, score = self.evaluate_document(query, doc)
            logger.info(f"  文档 {i+1}: quality={quality.value}, score={score}")

            if quality == RetrievalQuality.CORRECT:
                doc.metadata["crag_quality"] = "correct"
                doc.metadata["crag_score"] = score
                correct_docs.append(doc)
            elif quality == RetrievalQuality.AMBIGUOUS:
                doc.metadata["crag_quality"] = "ambiguous"
                doc.metadata["crag_score"] = score
                ambiguous_docs.append(doc)
            else:
                incorrect_count += 1

        # 整体判断
        total = len(documents)
        correct_ratio = len(correct_docs) / total

        if correct_ratio >= 0.5:
            overall = RetrievalQuality.CORRECT
            relevant_docs = correct_docs + ambiguous_docs  # 保留所有非 INCORRECT 的
        elif incorrect_count == total:
            overall = RetrievalQuality.INCORRECT
            relevant_docs = []
        else:
            overall = RetrievalQuality.AMBIGUOUS
            relevant_docs = correct_docs + ambiguous_docs

        logger.info(f"📋 整体判断: {overall.value} "
                     f"(correct={len(correct_docs)}, ambiguous={len(ambiguous_docs)}, "
                     f"incorrect={incorrect_count})")

        return overall, relevant_docs


# =============================================================================
# 第二部分：知识精炼（Knowledge Refinement）
# =============================================================================

class KnowledgeRefiner:
    """
    知识精炼器 - 对应论文中的 "Decompose-then-Recompose" 算法

    【工作原理】
    论文中的做法：
    1. Decompose（分解）: 把每个文档拆成更小的知识片段
    2. Filter（过滤）: 用 relevance evaluator 过滤掉不相关的片段
    3. Recompose（重组）: 把剩下的相关片段拼接成精炼后的上下文

    简化实现：用 LLM 直接提取文档中与查询相关的关键信息
    这比论文方法更简单但效果类似（面试时说明即可）
    """

    def __init__(self):
        self.api_key = config.DASHSCOPE_API_KEY
        self.model_name = config.CHAT_MODEL_NAME

    def refine(self, query: str, documents: list[Document]) -> str:
        """
        从文档中提取与查询相关的关键信息

        Args:
            query: 用户查询
            documents: 文档列表
        Returns:
            精炼后的上下文字符串
        """
        if not documents:
            return "未找到相关信息"

        # 拼接所有文档内容
        all_content = ""
        for i, doc in enumerate(documents, 1):
            all_content += f"\n--- 文档 {i} ---\n{doc.page_content[:400]}\n"

        refine_prompt = f"""请从以下文档中提取与查询最相关的关键信息。
去除无关内容，只保留对回答查询有帮助的核心信息。

【查询】{query}

【文档内容】{all_content}

【要求】
1. 只提取与查询直接相关的信息
2. 用简洁清晰的语言组织
3. 如果文档中没有相关信息，请明确说明

提取的关键信息:"""

        try:
            response = Generation.call(
                api_key=self.api_key,
                model=self.model_name,
                messages=[{"role": "user", "content": refine_prompt}],
                result_format="message",
            )

            if response.status_code == 200:
                return response.output.choices[0].message.get("content", "信息提取失败")

        except Exception as e:
            logger.warning(f"知识精炼失败: {e}")

        # 降级：直接拼接原始文档
        return all_content


# =============================================================================
# 第三部分：CRAG 完整管道（Corrective RAG Pipeline）
# =============================================================================

class CRAGPipeline:
    """
    CRAG 完整管道 - 把所有组件串联起来

    【完整流程】
    1. 混合检索 (Hybrid Search) → 召回候选文档
    2. 重排序 (Re-Ranking) → 精排候选文档
    3. 检索评估 (Retrieval Evaluator) → 判断文档质量
    4. 根据质量决定动作：
       - CORRECT → 知识精炼 → 生成回答
       - AMBIGUOUS → 知识精炼 + Web搜索 → 生成回答
       - INCORRECT → Web搜索 → 生成回答
    5. LLM 生成最终回答

    【面试话术】
    "我的 RAG 系统实现了完整的 CRAG 管道：
    首先用混合检索（BM25+向量+RRF）做粗筛，
    然后用 LLM Reranker 做精排，
    接着用检索评估器判断文档质量——
    如果相关就直接用精炼后的知识回答，
    如果不确定就同时用知识库和 Web 搜索，
    如果不相关就完全降级到 Web 搜索。
    这套自纠正机制让系统在知识库覆盖不全时也能给出准确回答。"
    """

    def __init__(self, documents: list[Document], vector_store):
        """
        Args:
            documents: 知识库文档列表
            vector_store: 向量数据库实例
        """
        # 初始化各组件
        self.hybrid_searcher = HybridSearchRetriever(documents, vector_store)
        self.reranker = BatchLLMReranker()
        self.evaluator = RetrievalEvaluator()
        self.refiner = KnowledgeRefiner()

        # Web 搜索工具（降级用）
        self.web_search = BaiduSearchTool(
            config.BAIDU_SEARCH_API_KEY,
            config.BAIDU_SEARCH_BASE_URL
        )

        self.api_key = config.DASHSCOPE_API_KEY
        self.model_name = config.CHAT_MODEL_NAME

        logger.info("CRAG Pipeline 初始化完成")

    def ask(self, query: str, status_filter: str = None) -> str:
        """
        CRAG 问答 - 完整的纠正式检索增强生成流程

        Args:
            query: 用户问题
            status_filter: 可选的元数据过滤
        Returns:
            AI 生成的回答
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 CRAG Pipeline 启动: query='{query}'")
        logger.info(f"{'='*60}")

        # ===== Step 1: 混合检索（粗筛）=====
        logger.info("\n📌 Step 1: 混合检索 (Hybrid Search)")
        retrieved_docs = self.hybrid_searcher.search(
            query, top_k=10, status_filter=status_filter
        )
        logger.info(f"  混合检索返回 {len(retrieved_docs)} 个文档")

        if not retrieved_docs:
            logger.info("  ⚠️ 检索无结果，直接降级到 Web 搜索")
            return self._generate_with_web_search(query)

        # ===== Step 2: 重排序（精排）=====
        logger.info("\n📌 Step 2: 重排序 (Re-Ranking)")
        reranked_docs = self.reranker.rerank(query, retrieved_docs, top_k=5)
        logger.info(f"  精排后保留 {len(reranked_docs)} 个文档")

        # ===== Step 3: 检索评估（CRAG 核心）=====
        logger.info("\n📌 Step 3: 检索评估 (Retrieval Evaluation)")
        quality, relevant_docs = self.evaluator.evaluate_documents(query, reranked_docs)

        # ===== Step 4: 根据质量执行不同策略 =====
        logger.info(f"\n📌 Step 4: 执行策略 → {quality.value}")

        if quality == RetrievalQuality.CORRECT:
            # ✅ 文档相关 → 精炼知识 → 直接生成
            logger.info("  ✅ CORRECT: 文档质量好，使用知识库回答")
            context = self.refiner.refine(query, relevant_docs)
            answer = self._generate_answer(query, context, source="知识库")

        elif quality == RetrievalQuality.AMBIGUOUS:
            # ⚠️ 不确定 → 知识库 + Web搜索 双重信息
            logger.info("  ⚠️ AMBIGUOUS: 补充 Web 搜索")
            kb_context = self.refiner.refine(query, relevant_docs)
            web_context = self._web_search(query)
            combined_context = f"【知识库信息】\n{kb_context}\n\n【Web搜索补充】\n{web_context}"
            answer = self._generate_answer(query, combined_context, source="知识库+Web搜索")

        else:
            # ❌ 不相关 → 完全降级到 Web 搜索
            logger.info("  ❌ INCORRECT: 知识库无相关信息，降级到 Web 搜索")
            answer = self._generate_with_web_search(query)

        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 CRAG Pipeline 完成")
        logger.info(f"{'='*60}")

        return answer

    def _web_search(self, query: str) -> str:
        """执行 Web 搜索获取外部知识"""
        logger.info(f"  🌐 Web 搜索: '{query}'")
        result = self.web_search.search(query)
        return result

    def _generate_with_web_search(self, query: str) -> str:
        """完全基于 Web 搜索结果生成回答"""
        web_context = self._web_search(query)
        return self._generate_answer(query, web_context, source="Web搜索")

    def _generate_answer(self, query: str, context: str, source: str = "unknown") -> str:
        """
        使用 LLM 基于上下文生成回答

        Args:
            query: 用户问题
            context: 检索/搜索得到的上下文
            source: 信息来源标注
        """
        prompt = f"""你是恋爱咨询助手。请基于以下参考信息回答用户问题。

【信息来源】{source}

【参考信息】
{context}

【用户问题】
{query}

【要求】
1. 基于参考信息给出专业、有针对性的回答
2. 如果参考信息不足以完全回答问题，请诚实说明
3. 回答要有条理、温暖、有建设性

你的回答:"""

        try:
            response = Generation.call(
                api_key=self.api_key,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                result_format="message",
            )

            if response.status_code == 200:
                answer = response.output.choices[0].message.get("content", "回答生成失败")
                logger.info(f"  💬 回答生成完成 (来源: {source})")
                return answer

        except Exception as e:
            logger.error(f"回答生成失败: {e}")

        return f"很抱歉，我暂时无法回答这个问题。(来源: {source})"
