"""
===============================================================================
🔍 Hybrid Search 混合检索 - 高级 RAG 技术 #1
===============================================================================

📚 论文/来源：工业界广泛采用的最佳实践，LangChain / LlamaIndex 原生支持

🧠 【核心概念 - 面试必备】

传统 RAG 只用向量语义检索（Dense Retrieval），有一个致命问题：
- 向量检索擅长理解"语义相似"，比如"如何追女生" ≈ "怎样追求女孩"
- 但它对"精确关键词"很弱，比如搜"BM25算法"可能检索不到包含"BM25"的文档

解决方案：Hybrid Search（混合检索）= 向量检索 + 关键词检索，两路并行，取长补短！

📐 【算法原理】

1. 向量检索（Dense Retrieval）：
   - 把文本通过 Embedding 模型转成向量
   - 用余弦相似度找最相似的文档
   - 优点：理解语义，"苹果手机" ≈ "iPhone"
   - 缺点：对精确术语、人名、产品名不敏感

2. BM25 关键词检索（Sparse Retrieval）：
   - 经典信息检索算法（基于 TF-IDF 的改进版）
   - 统计词频(TF)和逆文档频率(IDF)来打分
   - 优点：精确匹配关键词，速度快
   - 缺点：不理解语义，"汽车" ≠ "轿车"

3. RRF（Reciprocal Rank Fusion）融合排序：
   - 将两路结果按排名融合，而不是按分数融合
   - 公式: RRF_score(d) = Σ 1/(k + rank(d))，k 通常取 60
   - 为什么用排名而不用分数？因为两个系统的分数量纲不同，无法直接比较！

🔄 【完整流程图】

用户查询
    ├── [路径A] 向量检索 → 返回 Top-20（按语义相似度排序）
    ├── [路径B] BM25检索  → 返回 Top-20（按关键词匹配度排序）
    └── [RRF 融合] → 合并去重 → 按 RRF 分数重排 → 输出 Top-K

💡 【面试话术】
"为了解决纯语义检索对专有名词和精确查询的漏召回问题，
我引入了 BM25 + 向量的混合检索策略，使用 RRF（Reciprocal Rank Fusion）
进行融合排序。这种方案兼顾了语义理解和精确匹配，
显著提升了检索的召回率和准确率。"

📖 【学习路径】
1. 先理解 TF-IDF：https://zh.wikipedia.org/wiki/Tf-idf
2. 再理解 BM25：BM25 是 TF-IDF 的改进版，加入了文档长度归一化
3. 最后理解 RRF：https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
4. LangChain 的 EnsembleRetriever 文档
===============================================================================
"""

import logging
import math
from collections import defaultdict
from langchain_core.documents import Document
from langchain_chroma import Chroma
from rag.vector_store import get_retriever

logger = logging.getLogger(__name__)


# =============================================================================
# 第一部分：BM25 检索器（Sparse Retrieval）
# =============================================================================

class BM25Retriever:
    """
    BM25 关键词检索器

    BM25（Best Matching 25）是信息检索领域最经典的算法之一。
    它是 TF-IDF 的改进版本，主要改进了两个问题：
    1. TF（词频）饱和：当一个词出现 100 次 vs 10 次时，重要性不应该是 10 倍
       → BM25 用 tf / (tf + k1) 实现饱和效果，k1 通常取 1.2
    2. 文档长度归一化：长文档天然包含更多词，需要惩罚
       → BM25 用 (1 - b + b * dl/avgdl) 实现，b 通常取 0.75

    BM25 打分公式:
    score(q, d) = Σ IDF(qi) * tf(qi, d) * (k1 + 1) / (tf(qi, d) + k1 * (1 - b + b * |d|/avgdl))

    其中:
    - q 是查询，d 是文档
    - qi 是查询中的第 i 个词
    - tf(qi, d) 是词 qi 在文档 d 中的出现次数
    - IDF(qi) 是词 qi 的逆文档频率
    - |d| 是文档 d 的长度（词数）
    - avgdl 是所有文档的平均长度
    - k1 = 1.2（控制词频饱和度）
    - b = 0.75（控制文档长度归一化强度）
    """

    def __init__(self, documents: list[Document], k1: float = 1.2, b: float = 0.75):
        """
        初始化 BM25 检索器

        Args:
            documents: 文档列表
            k1: 词频饱和参数（越大，高频词得分越高，典型值 1.2-2.0）
            b: 文档长度归一化参数（0=不归一化，1=完全归一化，典型值 0.75）
        """
        self.documents = documents
        self.k1 = k1
        self.b = b

        # 预处理：对所有文档进行分词和统计
        self.doc_count = len(documents)  # 文档总数 N
        self.doc_lengths = []            # 每个文档的长度（词数）
        self.doc_term_freqs = []         # 每个文档中每个词的出现次数 tf(t, d)
        self.doc_freq = defaultdict(int) # 包含某个词的文档数 df(t)（用于计算 IDF）
        self.avg_doc_length = 0          # 所有文档的平均长度 avgdl

        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        """
        简单分词

        【注意】这里用的是最简单的按字和空格分词。
        实际生产中应该用 jieba 分词（中文）或 nltk/spacy（英文）。
        面试时可以提一下："我这里用了简单分词，生产环境可以换成 jieba 做更精准的中文分词"
        """
        tokens = []
        # 中文：逐字分词（简化版，实际应用 jieba）
        # 英文：按空格分词
        import re
        # 先按空格和标点分割
        words = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text.lower())
        return words

    def _build_index(self):
        """
        构建 BM25 索引

        预计算所有需要的统计量：
        1. 每个文档的词频表 (term frequency)
        2. 每个词的文档频率 (document frequency)
        3. 文档平均长度 (average document length)
        """
        total_length = 0

        for doc in self.documents:
            # 分词
            tokens = self._tokenize(doc.page_content)

            # 记录文档长度
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            # 统计词频 tf(t, d)
            term_freq = defaultdict(int)
            for token in tokens:
                term_freq[token] += 1
            self.doc_term_freqs.append(term_freq)

            # 统计文档频率 df(t)：有多少文档包含这个词
            seen_terms = set(tokens)
            for term in seen_terms:
                self.doc_freq[term] += 1

        # 计算平均文档长度
        self.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 0

        logger.info(f"BM25 索引构建完成: {self.doc_count} 个文档, 平均长度 {self.avg_doc_length:.1f} 词")

    def _compute_idf(self, term: str) -> float:
        """
        计算 IDF（逆文档频率）

        IDF 衡量一个词的"稀有程度"：
        - 如果一个词在几乎所有文档中都出现 → IDF 很低（不重要，如"的"、"是"）
        - 如果一个词只在少数文档中出现 → IDF 很高（很重要，如"BM25"、"向量检索"）

        公式: IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        其中 N 是文档总数，df(t) 是包含词 t 的文档数
        """
        df = self.doc_freq.get(term, 0)
        # 使用 Robertson-Sparck Jones IDF 公式（带平滑）
        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 20) -> list[tuple[Document, float]]:
        """
        执行 BM25 搜索

        Args:
            query: 用户查询文本
            top_k: 返回前 K 个最相关的文档
        Returns:
            [(Document, score), ...] 按分数降序排列
        """
        query_tokens = self._tokenize(query)
        scores = []

        for i, doc in enumerate(self.documents):
            score = 0.0
            doc_len = self.doc_lengths[i]
            term_freq = self.doc_term_freqs[i]

            for token in query_tokens:
                if token not in term_freq:
                    continue  # 该词不在此文档中，跳过

                tf = term_freq[token]       # 词频
                idf = self._compute_idf(token)  # 逆文档频率

                # BM25 核心公式
                # 分子: tf * (k1 + 1)
                numerator = tf * (self.k1 + 1)
                # 分母: tf + k1 * (1 - b + b * dl / avgdl)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                # 最终得分 = IDF * 分子/分母
                score += idf * numerator / denominator

            scores.append((doc, score))

        # 按分数降序排列，取 Top-K
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# =============================================================================
# 第二部分：RRF 融合排序（Reciprocal Rank Fusion）
# =============================================================================

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Document, float]]],
    k: int = 60,
    top_n: int = 10
) -> list[Document]:
    """
    RRF（Reciprocal Rank Fusion）融合排序算法

    📚 论文来源: "Reciprocal Rank Fusion outperforms Condorcet and individual
    Rank Learning Methods" (Cormack et al., 2009, SIGIR)

    🧠 【核心思想】
    把多个检索系统的结果按"排名"而不是"分数"来融合。
    为什么？因为不同系统的分数量纲不同（向量用余弦相似度 0-1，BM25 用对数分数 0-∞），
    直接比较分数没有意义，但"排名"是可以跨系统比较的！

    📐 【公式】
    对于文档 d，它在第 r 个检索系统中排名为 rank_r(d)，则：

    RRF_score(d) = Σ  1 / (k + rank_r(d))

    其中 k 是一个常数（通常 = 60），作用是平滑排名差异。
    - k 越大，排名靠前和靠后的文档分数差距越小（更平均）
    - k 越小，排名靠前的文档优势更大

    🌰 【举例】
    假设文档 A 在 BM25 中排第 1，在向量中排第 3：
    RRF(A) = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323

    假设文档 B 在 BM25 中排第 10，在向量中排第 2：
    RRF(B) = 1/(60+10) + 1/(60+2) = 0.0143 + 0.0161 = 0.0304

    结果: A 排在 B 前面 ✅（因为 A 在两个系统中都比较靠前）

    Args:
        ranked_lists: 多个检索系统的排序结果列表
                      每个列表是 [(Document, score), ...] 按分数降序
        k: RRF 常数，默认 60（来自原论文推荐值）
        top_n: 融合后返回前 N 个文档
    Returns:
        融合排序后的文档列表
    """
    # 用文档内容作为唯一标识（去重 key）
    rrf_scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, (doc, _score) in enumerate(ranked_list, start=1):
            # 用文档内容的前200字符作为去重 key（避免重复文档被算两次）
            doc_key = doc.page_content[:200]

            # RRF 公式：1 / (k + rank)
            rrf_scores[doc_key] += 1.0 / (k + rank)
            doc_map[doc_key] = doc  # 保存文档引用

    # 按 RRF 分数降序排列
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # 取 Top-N
    results = []
    for doc_key, score in sorted_docs[:top_n]:
        doc = doc_map[doc_key]
        # 把 RRF 分数写入 metadata，方便调试和面试展示
        doc.metadata["rrf_score"] = round(score, 6)
        results.append(doc)
        logger.info(f"  RRF 排名 #{len(results)}: score={score:.6f}, "
                     f"内容={doc.page_content[:50]}...")

    return results


# =============================================================================
# 第三部分：Hybrid Search 混合检索器（对外接口）
# =============================================================================

class HybridSearchRetriever:
    """
    混合检索器 - 组合 BM25 + 向量检索 + RRF 融合

    【使用方式】
    1. 初始化时传入文档列表和向量数据库
    2. 调用 search() 方法执行混合检索
    3. 内部自动完成：BM25检索 → 向量检索 → RRF融合 → 返回结果

    【面试场景】
    Q: "你上面说的混合检索具体是怎么实现的？"
    A: "我用 BM25 做关键词检索、用 DashScope Embedding + ChromaDB 做向量语义检索，
       两路各取 Top-20 候选文档，然后用 RRF 算法融合排序，最终取 Top-K。
       BM25 保证精确关键词不漏，向量保证语义相近的也能召回。"

    Q: "为什么用 RRF 而不是简单地合并？"
    A: "因为 BM25 和向量检索的分数量纲不同，一个是对数概率，一个是余弦相似度，
       不能直接比较分数。RRF 基于排名融合，是量纲无关的，而且在 SIGIR'09 论文中
       被证明效果优于其他融合方法。"
    """

    def __init__(self, documents: list[Document], vector_store: Chroma):
        """
        Args:
            documents: 原始文档列表（用于初始化 BM25）
            vector_store: 向量数据库实例（用于向量检索）
        """
        # 初始化 BM25 检索器
        self.bm25 = BM25Retriever(documents)
        # 保存向量数据库引用
        self.vector_store = vector_store
        # 保存文档引用
        self.documents = documents

        logger.info(f"混合检索器初始化完成: {len(documents)} 个文档")

    def search(self, query: str, top_k: int = 5, status_filter: str = None) -> list[Document]:
        """
        执行混合检索

        Args:
            query: 用户查询
            top_k: 最终返回的文档数
            status_filter: 可选的元数据过滤（如 "单身"）
        Returns:
            融合排序后的 Top-K 文档列表
        """
        logger.info(f"🔍 开始混合检索: query='{query}'")

        # ===== 路径 A：BM25 关键词检索 =====
        logger.info("  📝 执行 BM25 关键词检索...")
        bm25_results = self.bm25.search(query, top_k=20)
        logger.info(f"  📝 BM25 返回 {len(bm25_results)} 个结果")

        # ===== 路径 B：向量语义检索 =====
        logger.info("  🧮 执行向量语义检索...")
        search_kwargs = {"k": 20}
        if status_filter:
            search_kwargs["filter"] = {"status": status_filter}

        vector_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )
        vector_docs = vector_retriever.invoke(query)
        # 转换为 (doc, score) 格式（向量检索不返回分数，用排名模拟）
        vector_results = [(doc, 1.0 / (i + 1)) for i, doc in enumerate(vector_docs)]
        logger.info(f"  🧮 向量检索返回 {len(vector_results)} 个结果")

        # ===== RRF 融合排序 =====
        logger.info("  🔀 执行 RRF 融合排序...")
        fused_results = reciprocal_rank_fusion(
            ranked_lists=[bm25_results, vector_results],
            k=60,
            top_n=top_k,
        )

        logger.info(f"🔍 混合检索完成: 最终返回 {len(fused_results)} 个文档")
        return fused_results
