import logging
from typing import Any, Dict
from rag.vector_store import get_retriever
from rag.rag_chain import format_documents
from rag.query_rewriter import rewrite_query
import config

logger = logging.getLogger(__name__)

class InternalKnowledgeTool:
    """
    内部知识库查询工具
    允许 Agent 查询银行内部的理财产品、贷款政策等私密文档
    """
    
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def query_knowledge(self, query: str, category: str = "全部") -> str:
        """
        查询内部知识库获取详细政策或产品信息
        
        Args:
            query: 查询关键词或问题
            category: 类别过滤（"储蓄", "贷款", "投资", "全部"）
        """
        try:
            # 1. 查询改写（优化检索效果）
            rewritten_query = rewrite_query(query)
            
            # 2. 获取检索器
            status_filter = None if category == "全部" else category
            retriever = get_retriever(self.vector_store, status_filter=status_filter, top_k=3)
            
            # 3. 检索
            docs = retriever.invoke(rewritten_query)
            context = format_documents(docs)
            
            if not context:
                return "在内部知识库中未找到相关信息。"
            
            return f"从内部知识库检索到的相关内容如下：\n{context}"
            
        except Exception as e:
            logger.error(f"知识库查询失败: {e}")
            return f"查询内部知识库时出错: {str(e)}"

    def get_schema(self) -> Dict[str, Any]:
        """获取工具 Schema"""
        return {
            "name": "queryInternalKnowledge",
            "description": "查询银行内部私有知识库，包含理财产品细节、内部贷款政策、审批准则等非公开信息。当遇到通用大模型无法回答的银行特定业务问题时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "需要查询的关键词或具体问题"
                    },
                    "category": {
                        "type": "string",
                        "description": "知识类别",
                        "enum": ["储蓄", "贷款", "投资", "全部"],
                        "default": "全部"
                    }
                },
                "required": ["query"]
            }
        }
