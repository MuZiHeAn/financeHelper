"""
FinanceApp 金融理财助手应用 - 对应 Java LoveApp 改造版

集成了：
1. 基础对话（支持多轮对话记忆）
2. RAG 知识库问答
3. 工具调用

【对应底层逻辑】
- ChatClient.builder(model).defaultAdvisors(memory, log).build() → LangChain Chain + Memory
- MessageChatMemoryAdvisor → LangChain FileChatMessageHistory
- KryoFileChatMemory → 文件持久化对话记忆
- ReReadingAdvisor → 在 prompt 中实现 re-reading
"""
import os
import json
import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_chroma import Chroma
from rag.document_loader import load_markdown_documents
from rag.vector_store import create_vector_store, get_retriever
from rag.query_rewriter import rewrite_query
from rag.rag_chain import format_documents
import config

logger = logging.getLogger(__name__)

# 系统提示词 - 金融版
SYSTEM_PROMPT = (
    "你扮演深耕金融理财领域的专家。开场向用户表明身份，告知用户可咨询理财和金融难题。"
    "围绕储蓄、贷款、投资三种状态分析，储蓄状态关注存款利率和稳健理财，"
    "贷款状态关注房贷车贷政策及还款策略，投资状态关注股票基金及资产配置。"
    "引导用户详述资金状况、风险偏好及理财目标，以便给出专属解决方案。"
    "注意：所有涉及具体产品的建议必须包含免责声明：'投资有风险，入市需谨慎'。"
)


class FinanceApp:
    """
    金融理财专家应用
    
    功能：
    1. do_chat() - 基础多轮对话（带记忆）
    2. do_chat_with_rag() - RAG 知识库增强对话
    """

    def __init__(self):
        """初始化"""
        # 1. 创建对话模型
        self.chat_model = ChatTongyi(
            model=config.CHAT_MODEL_NAME,
            dashscope_api_key=config.DASHSCOPE_API_KEY,
        )

        # 2. 构建对话链（带记忆）
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("user", "{input}"),
        ])

        self.chain = self.prompt_template | self.chat_model | StrOutputParser()

        # 3. 包装为带历史记忆的链
        self.conversation_chain = RunnableWithMessageHistory(
            self.chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        # 4. 初始化向量存储（延迟加载）
        self._vector_store: Chroma | None = None

        logger.info("FinanceApp 初始化完成")

    def _get_session_history(self, session_id: str) -> FileChatMessageHistory:
        """获取会话历史（文件存储）"""
        os.makedirs(config.CHAT_MEMORY_DIR, exist_ok=True)
        file_path = os.path.join(config.CHAT_MEMORY_DIR, f"{session_id}.json")
        return FileChatMessageHistory(file_path)

    def _get_vector_store(self) -> Chroma:
        """延迟初始化向量存储"""
        if self._vector_store is None:
            documents = load_markdown_documents()
            if documents:
                self._vector_store = create_vector_store(documents)
            else:
                self._vector_store = create_vector_store()
        return self._vector_store

    def do_chat(self, message: str, chat_id: str) -> str:
        """
        基础对话（支持多轮记忆）

        Args:
            message: 用户消息
            chat_id: 对话 ID（区分不同用户/会话）
        Returns:
            AI 回复内容
        """
        session_config = {"configurable": {"session_id": chat_id}}

        try:
            result = self.conversation_chain.invoke(
                {"input": message},
                config=session_config,
            )
            logger.info(f"AI Response: {result[:100]}...")
            return result

        except Exception as e:
            logger.error(f"对话失败: {e}", exc_info=True)
            return f"对话出错: {e}"

    def do_chat_with_rag(self, message: str, chat_id: str, status_filter: str = "储蓄") -> str:
        """
        RAG 知识库增强对话

        流程：
        1. 查询改写
        2. 向量检索（带 status 过滤）
        3. 上下文增强
        4. LLM 回答

        Args:
            message: 用户消息
            chat_id: 对话 ID
            status_filter: 按状态过滤知识库（"储蓄"/"贷款"/"投资"）
        Returns:
            AI 回复内容
        """
        try:
            # 1. 查询改写
            rewritten_message = rewrite_query(message)

            # 2. 向量检索
            vector_store = self._get_vector_store()
            retriever = get_retriever(vector_store, status_filter=status_filter, top_k=3)
            docs = retriever.invoke(rewritten_message)
            context = format_documents(docs)

            # 3. 构建带上下文的提示
            rag_prompt = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT + f"\n\n参考资料:\n{context}"),
                MessagesPlaceholder("history"),
                ("user", "{input}"),
            ])

            # 4. 构建带记忆的 RAG 链
            rag_chain = rag_prompt | self.chat_model | StrOutputParser()
            rag_conversation = RunnableWithMessageHistory(
                rag_chain,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="history",
            )

            session_config = {"configurable": {"session_id": chat_id}}
            result = rag_conversation.invoke(
                {"input": message},
                config=session_config,
            )

            logger.info(f"RAG Response: {result[:100]}...")
            return result

        except Exception as e:
            logger.error(f"RAG 对话失败: {e}", exc_info=True)
            return f"RAG 对话出错: {e}"
