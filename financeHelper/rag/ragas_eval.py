"""
===============================================================================
📊 RAGAS 评估 - 高级 RAG 技术 #4（RAG 质量度量）
===============================================================================

📚 框架来源：
   RAGAS (Retrieval Augmented Generation Assessment)
   论文: "RAGAS: Automated Evaluation of Retrieval Augmented Generation"
   GitHub: https://github.com/explodinggradients/ragas

🧠 【核心概念 - 面试加分项】

做完 RAG 系统后，面试官一定会问：
"你怎么知道你的 RAG 效果好不好？有量化的评估指标吗？"

RAGAS 提供了 4 个核心指标来评估 RAG 系统的质量：

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  1. Faithfulness（忠实度）                                           │
│     回答是否忠于检索到的文档？有没有"编造"文档中没有的信息？            │
│     → 衡量"幻觉"程度                                                │
│     → 分数越高 = 幻觉越少                                            │
│                                                                     │
│  2. Answer Relevancy（回答相关性）                                   │
│     回答是否真正在回答用户的问题？                                     │
│     → 衡量回答与问题的匹配度                                         │
│     → 分数越高 = 回答越切题                                           │
│                                                                     │
│  3. Context Precision（上下文精度）                                  │
│     检索到的文档中，有多少是真正相关的？                               │
│     → 衡量检索的"准确率"                                             │
│     → 分数越高 = 检索越精准                                           │
│                                                                     │
│  4. Context Recall（上下文召回率）                                   │
│     回答问题需要的信息，有多少被检索到了？                             │
│     → 衡量检索的"完整性"                                             │
│     → 分数越高 = 检索越全面                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

💡 【面试话术】
"我用 RAGAS 框架建立了 RAG 评估流水线，
用 Faithfulness 衡量幻觉率、Answer Relevancy 衡量回答质量、
Context Precision/Recall 衡量检索质量。
每次迭代优化后都会跑一遍评估，量化对比改进效果。"

📖 【学习路径】
1. 理解四个指标的含义
2. 跑一遍评估看看结果
3. 对比基础 RAG vs 高级 RAG（Hybrid+Rerank+CRAG）的数值差异
===============================================================================
"""

import logging
import re
from dataclasses import dataclass
from dashscope import Generation
import config

logger = logging.getLogger(__name__)


@dataclass
class RAGEvalResult:
    """
    RAG 评估结果 - 包含 RAGAS 四大指标

    每个指标范围 0.0 ~ 1.0，越高越好
    """
    faithfulness: float       # 忠实度：回答是否忠于文档
    answer_relevancy: float   # 回答相关性：回答是否切题
    context_precision: float  # 上下文精度：检索的文档是否精准
    context_recall: float     # 上下文召回：需要的信息是否被检索到

    def summary(self) -> str:
        """格式化输出评估结果"""
        return (
            f"\n{'='*50}\n"
            f"📊 RAGAS 评估结果\n"
            f"{'='*50}\n"
            f"  忠实度 (Faithfulness):      {self.faithfulness:.2f}/1.00\n"
            f"  回答相关性 (Answer Relevancy): {self.answer_relevancy:.2f}/1.00\n"
            f"  上下文精度 (Context Precision): {self.context_precision:.2f}/1.00\n"
            f"  上下文召回 (Context Recall):    {self.context_recall:.2f}/1.00\n"
            f"{'='*50}\n"
            f"  综合得分: {self.overall_score():.2f}/1.00\n"
            f"{'='*50}"
        )

    def overall_score(self) -> float:
        """计算综合得分（四项平均）"""
        return (self.faithfulness + self.answer_relevancy +
                self.context_precision + self.context_recall) / 4


class RAGEvaluator:
    """
    RAG 评估器 - 使用 LLM 实现 RAGAS 指标评估

    【设计说明】
    原版 RAGAS 库有自己的实现，但依赖较多且配置复杂。
    这里用 LLM 直接评估四个指标，原理相同但更轻量。
    面试时说明："我参考 RAGAS 的评估维度，用 LLM-as-Judge 方式实现。"
    """

    def __init__(self):
        self.api_key = config.DASHSCOPE_API_KEY
        self.model_name = config.CHAT_MODEL_NAME

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str = None,
    ) -> RAGEvalResult:
        """
        评估一次 RAG 问答的质量

        Args:
            question: 用户原始问题
            answer: RAG 系统生成的回答
            contexts: 检索到的文档内容列表
            ground_truth: 标准答案（可选，用于计算 recall）
        Returns:
            RAGEvalResult 评估结果
        """
        logger.info(f"📊 开始 RAGAS 评估...")

        context_text = "\n".join(contexts) if contexts else "无检索上下文"

        # 指标1: Faithfulness（忠实度）
        faithfulness = self._evaluate_faithfulness(answer, context_text)

        # 指标2: Answer Relevancy（回答相关性）
        answer_relevancy = self._evaluate_answer_relevancy(question, answer)

        # 指标3: Context Precision（上下文精度）
        context_precision = self._evaluate_context_precision(question, contexts)

        # 指标4: Context Recall（上下文召回）
        context_recall = self._evaluate_context_recall(
            question, answer, context_text, ground_truth
        )

        result = RAGEvalResult(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
        )

        logger.info(result.summary())
        return result

    def _llm_score(self, prompt: str) -> float:
        """通用 LLM 评分方法，返回 0.0-1.0"""
        try:
            response = Generation.call(
                api_key=self.api_key,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                result_format="message",
            )
            if response.status_code == 200:
                text = response.output.choices[0].message.get("content", "5").strip()
                numbers = re.findall(r'\d+', text)
                if numbers:
                    score = min(max(int(numbers[0]), 1), 10)
                    return score / 10.0  # 归一化到 0-1
        except Exception as e:
            logger.warning(f"评分失败: {e}")
        return 0.5

    def _evaluate_faithfulness(self, answer: str, context: str) -> float:
        """
        评估忠实度：回答中的信息是否都来自上下文？

        【RAGAS 原始方法】
        1. 从回答中提取所有陈述（claims）
        2. 逐条检查每个陈述是否能从上下文中推导出来
        3. Faithfulness = 可推导的陈述数 / 总陈述数

        【简化实现】用 LLM 直接打分（LLM-as-Judge）
        """
        prompt = f"""请评估以下"回答"对"参考上下文"的忠实程度。

【参考上下文】
{context[:800]}

【回答】
{answer[:500]}

【评分标准】
- 10分: 回答完全基于上下文，没有编造任何信息
- 7-9分: 回答大部分基于上下文，有少量合理推理
- 4-6分: 回答部分基于上下文，部分信息来源不明
- 1-3分: 回答大量编造了上下文中没有的信息

只输出一个 1-10 的整数:"""
        return self._llm_score(prompt)

    def _evaluate_answer_relevancy(self, question: str, answer: str) -> float:
        """
        评估回答相关性：回答是否切中问题？

        【RAGAS 原始方法】
        1. 从回答反向生成可能的问题
        2. 计算生成的问题与原始问题的相似度
        3. 相似度越高 = 回答越相关

        【简化实现】用 LLM 直接打分
        """
        prompt = f"""请评估以下"回答"与"问题"的相关程度。

【问题】
{question}

【回答】
{answer[:500]}

【评分标准】
- 10分: 回答直接、完整地回答了问题
- 7-9分: 回答基本回答了问题，可能缺少部分细节
- 4-6分: 回答部分相关，但有偏题
- 1-3分: 回答与问题几乎无关

只输出一个 1-10 的整数:"""
        return self._llm_score(prompt)

    def _evaluate_context_precision(self, question: str, contexts: list[str]) -> float:
        """
        评估上下文精度：检索的文档中有多少是真正相关的？

        【RAGAS 原始方法】
        对每个检索到的文档片段判断是否与问题相关，
        计算 Precision = 相关文档数 / 总文档数

        【简化实现】让 LLM 评估整体检索质量
        """
        if not contexts:
            return 0.0

        context_summary = ""
        for i, ctx in enumerate(contexts[:5], 1):
            context_summary += f"\n文档{i}: {ctx[:200]}\n"

        prompt = f"""请评估以下检索文档与查询的整体精度。

【查询】
{question}

【检索到的文档】
{context_summary}

【评分标准】
- 10分: 所有文档都与查询高度相关
- 7-9分: 大部分文档相关，少数不太相关
- 4-6分: 约一半文档相关
- 1-3分: 大部分文档与查询不相关

只输出一个 1-10 的整数:"""
        return self._llm_score(prompt)

    def _evaluate_context_recall(
        self, question: str, answer: str, context: str, ground_truth: str = None
    ) -> float:
        """
        评估上下文召回：回答问题需要的信息是否被检索到了？

        【RAGAS 原始方法】
        需要 ground_truth（标准答案），检查标准答案中的信息是否在上下文中出现。
        如果没有标准答案，降级为检查回答中的信息是否在上下文中出现。
        """
        reference = ground_truth if ground_truth else answer

        prompt = f"""请评估以下上下文是否包含了回答问题所需的足够信息。

【问题】
{question}

【参考答案】
{reference[:300]}

【检索到的上下文】
{context[:800]}

【评分标准】
- 10分: 上下文完整包含了回答问题所需的所有信息
- 7-9分: 上下文包含了大部分必要信息
- 4-6分: 上下文包含了部分必要信息
- 1-3分: 上下文缺少大量必要信息

只输出一个 1-10 的整数:"""
        return self._llm_score(prompt)


# =============================================================================
# 便捷函数：一键评估
# =============================================================================

def quick_evaluate(question: str, answer: str, contexts: list[str]) -> RAGEvalResult:
    """
    快速评估一次 RAG 问答

    使用示例:
        result = quick_evaluate(
            question="单身怎么脱单？",
            answer="可以多参加社交活动...",
            contexts=["文档1内容...", "文档2内容..."]
        )
        print(result.summary())
    """
    evaluator = RAGEvaluator()
    return evaluator.evaluate(question, answer, contexts)
