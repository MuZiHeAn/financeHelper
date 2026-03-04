"""
BaseAgent 基类 - 对应 Java 的 BaseAgent.java

定义了 Agent 的通用属性、状态机和核心执行循环。
所有具体的 Agent 实现都应继承此类。

状态机: IDLE -> RUNNING -> (FINISHED | ERROR)

【对应 Java 源码】
- AgentState 枚举 → Python Enum
- BaseAgent 类 → 保留 run() / step() 循环框架
- streamRun() → Python 用生成器（yield）实现流式输出
"""
import logging
from enum import Enum
from abc import ABC, abstractmethod
from typing import Generator

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent 状态枚举 - 对应 Java AgentState.java"""
    IDLE = "IDLE"           # 空闲：未开始执行，等待 run() 被调用
    RUNNING = "RUNNING"     # 运行中：处于 ReAct 流程循环内
    FINISHED = "FINISHED"   # 已结束：达到终止条件或调用了终止工具
    ERROR = "ERROR"         # 异常：流程执行出现未捕获错误


class BaseAgent(ABC):
    """
    抽象的 Agent 基类 - 对应 Java BaseAgent.java

    核心属性：
    - name: Agent 名称
    - system_prompt: 系统提示词
    - next_step_prompt: 下一步引导提示词
    - state: 当前状态（状态机）
    - max_steps: 最大执行步数
    - current_step: 当前步数
    - message_list: 对话历史（上下文记忆）
    """

    def __init__(self, name: str = "BaseAgent", max_steps: int = 10):
        self.name = name
        self.system_prompt = ""
        self.next_step_prompt = ""
        self.state = AgentState.IDLE
        self.max_steps = max_steps
        self.current_step = 0
        self.message_list: list[dict] = []  # {"role": "user/assistant/tool", "content": "..."}

    def run(self, user_prompt: str) -> str:
        """
        【同步阻塞】运行 Agent - 对应 Java BaseAgent.run()

        流程：
        1. 校验状态 & 输入
        2. 设置状态为 RUNNING，将用户输入加入对话历史
        3. 循环执行 step()，直到 FINISHED 或达到 max_steps
        4. 返回所有步骤结果的组合字符串

        Args:
            user_prompt: 用户输入文本
        Returns:
            包含所有执行步骤结果的字符串
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")
        if not user_prompt or not user_prompt.strip():
            raise RuntimeError("Cannot run agent with empty user prompt")

        self.state = AgentState.RUNNING
        self.message_list.append({"role": "user", "content": user_prompt})

        results = []
        try:
            for i in range(self.max_steps):
                if self.state == AgentState.FINISHED:
                    break

                step_number = i + 1
                self.current_step = step_number
                logger.info(f"Executing step {step_number}/{self.max_steps}")

                step_result = self.step()
                result = f"Step {step_number}: {step_result}"
                results.append(result)

            if self.current_step >= self.max_steps:
                self.state = AgentState.FINISHED
                results.append(f"Terminated: Reached max steps ({self.max_steps})")

            return "\n".join(results)

        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"Error executing agent: {e}", exc_info=True)
            return f"执行错误: {e}"
        finally:
            self.cleanup()

    def stream_run(self, user_prompt: str) -> Generator[str, None, None]:
        """
        【流式输出】运行 Agent - 对应 Java BaseAgent.streamRun()

        使用 Python 生成器（yield）代替 Java 的 Flux<String>，
        每执行完一步就 yield 当前结果，实现流式输出。

        Args:
            user_prompt: 用户输入文本
        Yields:
            每一步的执行结果字符串
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")
        if not user_prompt or not user_prompt.strip():
            raise RuntimeError("Cannot run agent with empty user prompt")

        self.state = AgentState.RUNNING
        self.message_list.append({"role": "user", "content": user_prompt})

        try:
            for i in range(self.max_steps):
                if self.state == AgentState.FINISHED:
                    break

                step_number = i + 1
                self.current_step = step_number
                step_header = f"## 正在执行第 {step_number}/{self.max_steps} 步\n"
                logger.info(step_header)
                yield step_header

                step_result = self.step()
                yield step_result + "\n\n"

            if self.current_step >= self.max_steps:
                self.state = AgentState.FINISHED
                yield f"已达到最大步骤数 ({self.max_steps})，任务终止。"

        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"代理执行出错: {e}", exc_info=True)
            yield f"执行错误: {e}"
        finally:
            self.cleanup()

    @abstractmethod
    def step(self) -> str:
        """
        Agent 的单个执行步骤 - 抽象方法

        由子类实现，封装 Agent 的核心决策和行动逻辑。
        Returns:
            当前步骤的执行结果字符串
        """
        pass

    def cleanup(self):
        """
        清理资源 - 对应 Java BaseAgent.cleanup()

        执行结束后重置状态，清空消息历史。
        """
        self.state = AgentState.IDLE
        self.current_step = 0
        self.message_list.clear()
