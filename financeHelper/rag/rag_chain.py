"""
RAG 检索增强链 - 对应 Java createLoveAppRagCustomAdvisorFactory.java

构建完整的 RAG 问答链：
  用户查询 → 查询改写 → 向量检索 → 上下文增强 → LLM 回答

【对应 Java 逻辑】
- RetrievalAugmentationAdvisor → LangChain RetrievalQA 链
- ContextualQueryAugmenter → 自定义 prompt template
- VectorStoreDocumentRetriever → ChromaDB retriever
"""
import logging
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_chroma import Chroma
from rag.vector_store import get_retriever
from rag.query_rewriter import rewrite_query
import config

logger = logging.getLogger(__name__)

# RAG 提示词模板 - 对应 Java 中的 ContextualQueryAugmenter 自定义模板
RAG_PROMPT_TEMPLATE = """你是恋爱咨询助手。请基于提供的上下文回答用户问题。

[上下文]
{context}

[问题]
{question}

[要求]
只基于上下文作答，若无相关信息请明确说明。"""


def format_documents(docs: list[Document]) -> str:
    """将文档列表格式化为上下文字符串"""
    if not docs:
        return "无相关参考资料"

    formatted = ""
    for doc in docs:
        formatted += f"文档片段：{doc.page_content}\n文档元数据：{doc.metadata}\n\n"
    return formatted


def create_rag_chain(vector_store: Chroma, status_filter: str = "单身"):
    """
    创建 RAG 问答链 - 对应 Java createLoveAppRagCustomAdvisor()

    构建 LangChain 链式调用：
    query → retriever → format → prompt → LLM → output

    Args:
        vector_store: ChromaDB 向量存储实例
        status_filter: 按 status 过滤文档（默认"单身"）
    Returns:
        可调用的 RAG Chain
    """
    # 1. 创建检索器（带 status 过滤）
    retriever = get_retriever(vector_store, status_filter=status_filter, top_k=3)

    # 2. 创建对话模型
    chat_model = ChatTongyi(
        model=config.CHAT_MODEL_NAME,
        dashscope_api_key=config.DASHSCOPE_API_KEY,
    )

    # 3. 创建提示词模板
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    # 4. 构建 RAG Chain
    rag_chain = (
        {
            "context": RunnableLambda(lambda x: x) | retriever | format_documents,
            "question": RunnablePassthrough(),
        }
        | prompt
        | chat_model
        | StrOutputParser()
    )

    return rag_chain


def ask_with_rag(vector_store: Chroma, question: str, status_filter: str = "单身") -> str:
    """
    使用 RAG 回答问题 - 便捷方法

    Args:
        vector_store: 向量存储
        question: 用户问题
        status_filter: 状态过滤
    Returns:
        AI 回答文本
    """
    # 1. 查询改写
    rewritten = rewrite_query(question)

    # 2. 调用 RAG Chain
    chain = create_rag_chain(vector_store, status_filter)
    result = chain.invoke(rewritten)

    return result
