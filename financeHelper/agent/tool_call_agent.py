"""
ToolCallAgent - 对应 Java 的 ToolCallAgent.java

实现了 ReAct 模式中的 think() 和 act() 方法。
使用 DashScope 的通义千问模型进行工具调用（Function Calling / Tool Calls）。

【核心工作流程】
think():
  1. 构建对话历史 + 工具列表
  2. 调用大模型，获取思考结果和工具调用决策
  3. 如果模型返回 tool_calls → return True（需要行动）
  4. 如果模型直接回复 → return False（无需行动）

act():
  1. 遍历模型返回的 tool_calls
  2. 在已注册的工具中查找并执行对应函数
  3. 将工具结果作为 tool 消息加入对话历史
  4. 检查是否调用了终止工具
"""
import json
import logging
from typing import Callable
from agent.react_agent import ReActAgent
from agent.base_agent import AgentState

logger = logging.getLogger(__name__)

# DashScope SDK
from dashscope import Generation


class ToolCallAgent(ReActAgent):
    """
    工具调用 Agent - 对应 Java ToolCallAgent.java

    关键属性：
    - available_tools: 可用工具字典 {name: {"function": callable, "description": ..., "parameters": ...}}
    - tool_calls: think() 阶段模型返回的工具调用列表（暂存，供 act() 使用）
    - api_key: DashScope API Key
    - model_name: 模型名称
    """

    def __init__(self, tools: list[dict], api_key: str, model_name: str = "qwen-turbo"):
        """
        Args:
            tools: 工具定义列表，每个工具是一个 dict:
                {
                    "type": "function",
                    "function": {
                        "name": "工具名",
                        "description": "工具描述",
                        "parameters": { JSON Schema }
                    }
                }
            api_key: DashScope API Key
            model_name: 使用的模型名称
        """
        super().__init__()
        self.tool_definitions = tools      # 工具定义（给模型看的 JSON Schema）
        self.tool_functions: dict[str, Callable] = {}  # 工具实际执行函数 {name: callable}
        self.tool_calls = []               # 暂存 think() 阶段的工具调用
        self.api_key = api_key
        self.model_name = model_name

    def register_tool_function(self, name: str, func: Callable):
        """注册工具的实际执行函数"""
        self.tool_functions[name] = func

    def think(self) -> bool:
        """
        思考阶段 - 对应 Java ToolCallAgent.think()

        1. 如果有 next_step_prompt，加入对话历史引导模型决策
        2. 构建完整的消息列表 + 系统提示
        3. 调用 DashScope 大模型（带工具定义）
        4. 分析返回：是否包含 tool_calls
        """
        # 1. 加入下一步引导提示
        if self.next_step_prompt:
            self.message_list.append({"role": "user", "content": self.next_step_prompt})

        # 2. 构建消息（加上 system prompt）
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.message_list)

        try:
            # 3. 调用 DashScope API（带工具调用）
            response = Generation.call(
                api_key=self.api_key,
                model=self.model_name,
                messages=messages,
                tools=self.tool_definitions if self.tool_definitions else None,
                result_format="message",
            )

            if response.status_code != 200:
                error_msg = f"模型调用失败: {response.code} - {response.message}"
                logger.error(error_msg)
                self.message_list.append({"role": "assistant", "content": error_msg})
                return False

            # 4. 解析模型返回
            assistant_message = response.output.choices[0].message

            # 获取文本内容
            content = assistant_message.get("content", "") or ""
            # 获取工具调用
            self.tool_calls = assistant_message.get("tool_calls", []) or []

            logger.info(f"{self.name} 的思考: {content}")
            logger.info(f"{self.name} 选择了 {len(self.tool_calls)} 个工具来使用")

            if self.tool_calls:
                for tc in self.tool_calls:
                    func_info = tc.get("function", {})
                    logger.info(f"  工具名称：{func_info.get('name')}，参数：{func_info.get('arguments')}")

                # 将完整的 assistant 消息（含 tool_calls）加入历史
                self.message_list.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": self.tool_calls
                })
                return True  # 需要行动
            else:
                # 模型直接回复，不需要工具
                self.message_list.append({"role": "assistant", "content": content})
                return False

        except Exception as e:
            logger.error(f"{self.name} 的思考过程遇到了问题: {e}", exc_info=True)
            self.message_list.append({"role": "assistant", "content": f"处理时遇到错误: {e}"})
            return False

    def act(self) -> str:
        """
        行动阶段 - 对应 Java ToolCallAgent.act()

        1. 遍历 tool_calls 列表
        2. 从已注册的工具函数中查找并执行
        3. 将每个工具的执行结果加入对话历史（role: tool）
        4. 检查是否调用了终止工具（doTerminate）
        """
        if not self.tool_calls:
            return "没有工具调用"

        results = []

        for tool_call in self.tool_calls:
            func_info = tool_call.get("function", {})
            tool_name = func_info.get("name", "")
            tool_args_str = func_info.get("arguments", "{}")
            tool_call_id = tool_call.get("id", "")

            # 解析参数
            try:
                tool_args = json.loads(tool_args_str) if isinstance(tool_args_str, str) else tool_args_str
            except json.JSONDecodeError:
                tool_args = {}

            # 查找并执行工具
            if tool_name in self.tool_functions:
                try:
                    tool_result = self.tool_functions[tool_name](**tool_args)
                except Exception as e:
                    tool_result = f"工具执行错误: {e}"
            else:
                tool_result = f"未找到工具: {tool_name}"

            # 将工具结果加入对话历史
            self.message_list.append({
                "role": "tool",
                "content": str(tool_result),
                "name": tool_name,
                "tool_call_id": tool_call_id
            })

            result_msg = f"工具 {tool_name} 完成了它的任务！结果: {tool_result}"
            results.append(result_msg)
            logger.info(result_msg)

            # 检查是否调用了终止工具
            if tool_name == "doTerminate":
                self.state = AgentState.FINISHED

        # 清空暂存的 tool_calls
        self.tool_calls = []

        return "\n".join(results)
