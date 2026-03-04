"""
终止工具 - 对应 Java TerminateTool.java

Agent 调用此工具表示任务已完成，触发 ReAct 循环终止。
"""


class TerminateTool:
    """终止工具 - 对应 Java TerminateTool"""

    def do_terminate(self) -> str:
        """终止交互 - 对应 Java @Tool doTerminate"""
        return "任务结束"

    def terminate_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "doTerminate",
                "description": (
                    "Terminate the interaction when the request is met OR if the assistant "
                    "cannot proceed further with the task. "
                    "When you have finished all the tasks, call this tool to end the work."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
