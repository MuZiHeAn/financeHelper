"""
向量存储配置 - 对应 Java LoveAppVectorStoreConfig.java + MyKeywordEnricher.java

使用 ChromaDB 作为本地向量数据库（代替 Java 的 SimpleVectorStore / PgVector）。
使用 DashScope Embedding 模型生成向量。

【对应 Java 逻辑】
- SimpleVectorStore.builder(embeddingModel).build() → ChromaDB
- myKeywordEnricher.enrichDocuments(docs) → 关键词提取增强 metadata
- simpleVectorStore.doAdd(documents) → chroma.add_documents()
"""
import os
import logging
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
import config

logger = logging.getLogger(__name__)


def create_embedding_model() -> DashScopeEmbeddings:
    """创建 DashScope 嵌入模型 - 对应 Java 的 dashscopeEmbeddingModel"""
    return DashScopeEmbeddings(
        model=config.EMBEDDING_MODEL_NAME,
        dashscope_api_key=config.DASHSCOPE_API_KEY,
    )


def enrich_keywords(documents: list[Document]) -> list[Document]:
    """
    关键词提取增强 - 对应 Java MyKeywordEnricher.enrichDocuments()

    从文档内容中提取关键词并添加到 metadata 中，
    增强后续的检索效果。
    """
    for doc in documents:
        content = doc.page_content
        # 简单的关键词提取：取前几个非空行作为关键词
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        keywords = []
        for line in lines[:3]:
            # 去掉 Markdown 标记
            cleaned = line.lstrip("#").strip()
            if cleaned and len(cleaned) < 50:
                keywords.append(cleaned)
        doc.metadata["keywords"] = ", ".join(keywords) if keywords else ""

    return documents


def create_vector_store(documents: list[Document] = None) -> Chroma:
    """
    创建向量存储 - 对应 Java LoveAppVectorStoreConfig.loveAppVectorStore()

    流程：
    1. 创建嵌入模型
    2. 创建或加载 ChromaDB 实例
    3. 如果提供了文档，添加到向量库

    Args:
        documents: 待存储的文档列表（None 则只加载已有数据库）
    Returns:
        ChromaDB 向量存储实例
    """
    embedding = create_embedding_model()

    # 确保持久化目录存在
    os.makedirs(config.CHROMA_PERSIST_DIR, exist_ok=True)

    vector_store = Chroma(
        collection_name=config.CHROMA_COLLECTION_NAME,
        embedding_function=embedding,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )

    if documents:
        # 关键词增强
        documents = enrich_keywords(documents)
        # 添加文档到向量库
        vector_store.add_documents(documents)
        logger.info(f"已将 {len(documents)} 个文档添加到向量库")

    return vector_store


def get_retriever(vector_store: Chroma, status_filter: str = None, top_k: int = 3):
    """
    创建检索器 - 对应 Java VectorStoreDocumentRetriever

    Args:
        vector_store: 向量存储实例
        status_filter: 按 status 元数据过滤（如 "单身"）
        top_k: 返回最相似的文档数
    Returns:
        LangChain Retriever 实例
    """
    search_kwargs = {"k": top_k}

    if status_filter:
        search_kwargs["filter"] = {"status": status_filter}

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )
