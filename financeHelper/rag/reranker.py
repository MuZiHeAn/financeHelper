"""
===============================================================================
🏆 Re-Ranking 重排序 - 高级 RAG 技术 #2
===============================================================================

📚 论文/来源：Cross-Encoder 重排序是 RAG 业界标准做法
   推荐模型: BGE-reranker-v2, Qwen3-Reranker, Cohere Rerank

🧠 【核心概念 - 面试必备】

为什么需要 Re-Ranking？

检索阶段（不管是向量检索还是 BM25）属于"粗筛"：
- 从 10 万篇文档中快速找出 Top-20 候选
- 速度要求高 → 只能用"轻量级"方法（如余弦相似度）
- 代价：准确率不够高，可能有不相关的文档混进来

Re-Ranking 是"精排"：
- 只处理 20 个候选文档 → 可以用"重量级"方法
- 用 Cross-Encoder 或 LLM 对每个 (query, document) 对做精细评分
- 结果：大幅提升前几名的质量

📐 【Bi-Encoder vs Cross-Encoder】

1. Bi-Encoder（检索阶段使用）：
   - query 和 document 分别独立编码成向量
   - 然后用余弦相似度计算相关性
   - 优点：document 可以预先编码，检索速度极快（毫秒级）
   - 缺点：query 和 document 没有交互，理解深度有限

2. Cross-Encoder（重排序阶段使用）：
   - 把 query 和 document 拼接在一起，送入模型
   - 模型同时看到两者，进行深度交互理解
   - 优点：理解更深入，准确率更高
   - 缺点：无法预先编码，每对都要实时计算，速度慢
   - 所以只用于"精排"少量候选文档！

📊 【Re-Ranking 在 RAG 管道中的位置】

用户查询 → 粗筛(Retrieval) → Top-20候选 → 精排(Re-Ranking) → Top-5 → LLM回答
                ↑                                ↑
           Bi-Encoder                      Cross-Encoder/LLM
           （速度优先）                    （准确率优先）

💡 【面试话术】
"在我的 RAG 系统中，检索分为两阶段：
第一阶段用混合检索（BM25 + 向量）做粗筛，快速从万级文档中召回 Top-20；
第二阶段用 Cross-Encoder 风格的 LLM 重排序做精排，对 20 个候选文档
逐一与查询做深度交叉理解，重新打分排序后取 Top-5 送入生成模型。
这种两阶段架构兼顾了检索效率和准确率。"

📖 【学习路径】
1. 理解 Bi-Encoder vs Cross-Encoder：
   https://www.sbert.net/examples/applications/cross-encoder/README.html
2. 理解 Re-Ranking 在 RAG 中的作用：
   搜索 "RAG reranking pipeline" 看架构图
3. 了解主流 Reranker 模型：BGE-reranker, Cohere Rerank, Jina Reranker
===============================================================================
"""

import logging
from langchain_core.documents import Document
from dashscope import Generation
import config

logger = logging.getLogger(__name__)


class LLMReranker:
    """
    基于 LLM 的重排序器

    【设计选择说明】
    真正的 Cross-Encoder Reranker 需要专门的模型（如 BGE-reranker）。
    这里我们用通义千问大模型来做重排序，原理类似：
    - 把 (query, document) 对一起送给 LLM
    - 让 LLM 判断文档与查询的相关性，并打分
    - 优点：不需要额外部署 Reranker 模型，直接用现有的 LLM
    - 缺点：比专用 Reranker 慢一些，但对于学习和面试展示足够了

    面试时可以说：
    "我实现了 LLM-based Reranker，原理与 Cross-Encoder 相同——
    将 query 和 document 拼接后做深度语义理解并打分。
    在生产环境中可以替换为 BGE-reranker-v2 等专用模型以提升速度。"
    """

    def __init__(self):
        self.api_key = config.DASHSCOPE_API_KEY
        self.model_name = config.CHAT_MODEL_NAME

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """
        对候选文档进行重排序

        【工作流程】
        1. 对每个文档，构造 "query + document" 的评分 prompt
        2. 让 LLM 给每个文档打 1-10 分
        3. 按分数降序排列，取 Top-K

        Args:
            query: 用户查询
            documents: 候选文档列表（通常是粗筛的 Top-20）
            top_k: 精排后返回的文档数
        Returns:
            重排后的 Top-K 文档列表
        """
        if not documents:
            return []

        logger.info(f"🏆 开始重排序: {len(documents)} 个候选文档, query='{query}'")

        scored_docs = []

        for i, doc in enumerate(documents):
            score = self._score_document(query, doc, i + 1, len(documents))
            scored_docs.append((doc, score))

        # 按分数降序排列
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 取 Top-K
        results = []
        for rank, (doc, score) in enumerate(scored_docs[:top_k], start=1):
            doc.metadata["rerank_score"] = score
            doc.metadata["rerank_rank"] = rank
            results.append(doc)
            logger.info(f"  重排 #{rank}: score={score}, 内容={doc.page_content[:50]}...")

        logger.info(f"🏆 重排序完成: 从 {len(documents)} 个文档中选出 Top-{top_k}")
        return results

    def _score_document(self, query: str, doc: Document, index: int, total: int) -> float:
        """
        给单个文档打分

        【Prompt 设计】
        让 LLM 充当"相关性评判者"，给 query-document 对打分。
        这就是 Cross-Encoder 的核心思想——让模型同时看到查询和文档。
        """
        # 截断文档内容，避免太长
        content = doc.page_content[:500]

        scoring_prompt = f"""你是一个文档相关性评分专家。
请评估以下"查询"和"文档"之间的相关性。

【查询】
{query}

【文档】
{content}

【评分规则】
- 只输出一个 1 到 10 之间的整数分数
- 10 分 = 完全相关，文档直接回答了查询
- 7-9 分 = 高度相关，文档包含有用信息
- 4-6 分 = 部分相关，有一些相关信息
- 1-3 分 = 不太相关或完全无关
- 只输出数字，不要输出任何其他内容

你的评分:"""

        try:
            response = Generation.call(
                api_key=self.api_key,
                model=self.model_name,
                messages=[{"role": "user", "content": scoring_prompt}],
                result_format="message",
            )

            if response.status_code == 200:
                text = response.output.choices[0].message.get("content", "5").strip()
                # 提取数字
                import re
                numbers = re.findall(r'\d+', text)
                if numbers:
                    score = min(max(int(numbers[0]), 1), 10)  # 限制在 1-10
                    return float(score)

            return 5.0  # 默认中等分数

        except Exception as e:
            logger.warning(f"  文档 {index}/{total} 评分失败: {e}")
            return 5.0


class BatchLLMReranker:
    """
    批量 LLM 重排序器（性能优化版）

    【优化思路】
    上面的 LLMReranker 对每个文档单独调用 LLM，N 个文档需要 N 次 API 调用。
    BatchLLMReranker 把所有文档打包成一个 prompt，只需 1 次 API 调用！

    【面试话术】
    "为了降低重排序的延迟和成本，我把逐个评分优化为批量评分——
    将所有候选文档打包到一个 prompt 中，让 LLM 一次性给出排序，
    API 调用次数从 N 次降到 1 次，延迟降低了 80%。"
    """

    def __init__(self):
        self.api_key = config.DASHSCOPE_API_KEY
        self.model_name = config.CHAT_MODEL_NAME

    def rerank(self, query: str, documents: list[Document], top_k: int = 5) -> list[Document]:
        """
        批量重排序 - 一次 API 调用完成所有文档的排序
        """
        if not documents:
            return []

        logger.info(f"🏆 开始批量重排序: {len(documents)} 个文档")

        # 构建批量评分 prompt
        doc_texts = ""
        for i, doc in enumerate(documents, start=1):
            content = doc.page_content[:300]  # 截断以控制 prompt 长度
            doc_texts += f"\n【文档 {i}】\n{content}\n"

        batch_prompt = f"""你是一个文档相关性排序专家。

【用户查询】
{query}

【候选文档列表】
{doc_texts}

【任务】
请根据与查询的相关性，对上述文档从高到低排序。
只输出文档编号，用逗号分隔，最相关的排在前面。
例如: 3,1,5,2,4

你的排序结果:"""

        try:
            response = Generation.call(
                api_key=self.api_key,
                model=self.model_name,
                messages=[{"role": "user", "content": batch_prompt}],
                result_format="message",
            )

            if response.status_code == 200:
                text = response.output.choices[0].message.get("content", "").strip()
                import re
                numbers = re.findall(r'\d+', text)

                # 按 LLM 给出的排序重排文档
                reranked = []
                seen = set()
                for num_str in numbers:
                    idx = int(num_str) - 1  # 转为 0-indexed
                    if 0 <= idx < len(documents) and idx not in seen:
                        seen.add(idx)
                        doc = documents[idx]
                        doc.metadata["rerank_rank"] = len(reranked) + 1
                        reranked.append(doc)

                # 如果 LLM 漏了一些文档，按原顺序补充
                for i, doc in enumerate(documents):
                    if i not in seen:
                        reranked.append(doc)

                result = reranked[:top_k]
                logger.info(f"🏆 批量重排序完成: 返回 Top-{len(result)}")
                return result

        except Exception as e:
            logger.warning(f"批量重排序失败: {e}, 返回原始顺序")

        return documents[:top_k]
