"""
配置管理模块 - 对应 Java 中的 application-local.yml + FileConstant

所有配置集中管理，通过 .env 文件加载敏感信息
"""
import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# ===== DashScope API 配置 =====
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "qwen-turbo")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v2")

# ===== 百度搜索 API 配置 =====
BAIDU_SEARCH_API_KEY = os.getenv("BAIDU_SEARCH_API_KEY", "")
BAIDU_SEARCH_BASE_URL = os.getenv("BAIDU_SEARCH_BASE_URL", "https://www.searchapi.io/api/v1/search")

# ===== 文件存储配置（对应 Java FileConstant）=====
FILE_SAVE_DIR = os.getenv("FILE_SAVE_DIR", "./temp")

# ===== 向量数据库配置 =====
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "love_app_knowledge"

# ===== RAG 文本处理配置 =====
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", "。", "；"]

# ===== Agent 配置 =====
AGENT_MAX_STEPS = 20

# ===== 知识库文档路径 =====
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")

# ===== 对话记忆配置 =====
CHAT_MEMORY_DIR = os.path.join(FILE_SAVE_DIR, "chat-memory")
