"""
查询改写器 - 对应 Java QueryRewriter.java

使用 LLM 对用户查询进行改写优化，提高 RAG 检索的召回率。

【对应 Java 逻辑】
- RewriteQueryTransformer → 用 LLM Chain 实现查询改写
- 清理思考块 <think>...</think> 和占位符符号
"""
import re
import logging
from dashscope import Generation
import config

logger = logging.getLogger(__name__)


def rewrite_query(prompt: str) -> str:
    """
    查询改写 - 对应 Java QueryRewriter.doQueryRewriter()

    使用大模型将用户的口语化查询改写为更适合检索的形式。

    Args:
        prompt: 用户原始查询
    Returns:
        改写后的查询文本
    """
    rewrite_system_prompt = """你是一个查询改写助手。
请将用户的问题改写为更清晰、更适合知识库检索的形式。
只输出改写后的查询，不要添加任何解释。"""

    try:
        response = Generation.call(
            api_key=config.DASHSCOPE_API_KEY,
            model=config.CHAT_MODEL_NAME,
            messages=[
                {"role": "system", "content": rewrite_system_prompt},
                {"role": "user", "content": f"请改写以下查询：{prompt}"},
            ],
            result_format="message",
        )

        if response.status_code != 200:
            logger.warning(f"查询改写失败，使用原始查询: {response.message}")
            return prompt

        text = response.output.choices[0].message.get("content", prompt)

        # 清理思考块和占位符（对应 Java 中的清理逻辑）
        text = re.sub(r"(?s)<think>.*?</think>", "", text)
        text = text.replace("<", "＜").replace(">", "＞")
        text = text.replace("{", "｛").replace("}", "｝")

        logger.info(f"查询改写: '{prompt}' → '{text}'")
        return text.strip()

    except Exception as e:
        logger.warning(f"查询改写异常，使用原始查询: {e}")
        return prompt
