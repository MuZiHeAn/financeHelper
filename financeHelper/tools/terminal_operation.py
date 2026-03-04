"""
终端操作工具 - 对应 Java TerminalOperationTool.java

使用 subprocess 代替 Java 的 ProcessBuilder。
"""
import subprocess
import logging

logger = logging.getLogger(__name__)


class TerminalOperationTool:
    """终端操作工具 - 对应 Java TerminalOperationTool"""

    def execute_terminal_command(self, command: str) -> str:
        """执行终端命令 - 对应 Java @Tool executeTerminalCommand"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
                encoding="utf-8",
                errors="replace"
            )
            output = result.stdout
            if result.returncode != 0:
                output += f"\nCommand execution failed with exit code: {result.returncode}"
                if result.stderr:
                    output += f"\nStderr: {result.stderr}"
            return output if output else "Command executed successfully (no output)"

        except subprocess.TimeoutExpired:
            return "Error: Command execution timed out (60s)"
        except Exception as e:
            return f"Error executing command: {e}"

    def execute_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "executeTerminalCommand",
                "description": "Execute a command in the terminal",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Command to execute in the terminal"}
                    },
                    "required": ["command"]
                }
            }
        }
