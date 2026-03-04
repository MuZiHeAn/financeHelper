"""
ReActAgent - 对应 Java 的 ReActAgent.java

ReAct = Reasoning（推理） + Acting（行动）
实现了 think/act/step 的模板方法模式。

工作流程：
step() → think() → 是否需要行动？ → act() → 返回结果
"""
import logging
from abc import abstractmethod
from agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ReActAgent(BaseAgent):
    """
    ReAct Agent 抽象类 - 对应 Java ReActAgent.java

    定义了 think() 和 act() 两个抽象方法，
    step() 中按照 ReAct 范式先思考再行动。
    """

    @abstractmethod
    def think(self) -> bool:
        """
        思考阶段 - 决定是否需要调用工具

        Returns:
            True: 需要执行 act()（调用工具）
            False: 不需要行动（直接回复或已完成）
        """
        pass

    @abstractmethod
    def act(self) -> str:
        """
        行动阶段 - 执行工具调用

        Returns:
            工具执行结果的描述字符串
        """
        pass

    def step(self) -> str:
        """
        执行单步 - 对应 Java ReActAgent.step()

        ReAct 核心流程：
        1. think() → 决策是否需要行动
        2. 如果不需要 → 返回"思考完成"
        3. 如果需要 → 调用 act() 并返回结果
        """
        try:
            should_act = self.think()  # 第一步：先思考
            if not should_act:
                return "思考完成 - 无需行动"
            return self.act()  # 进入行动阶段
        except Exception as e:
            logger.error(f"步骤执行失败: {e}", exc_info=True)
            return f"步骤执行失败: {e}"
