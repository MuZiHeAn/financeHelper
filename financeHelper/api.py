import logging
import json
import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from app.finance_app import FinanceApp
from agent.yu_manus import create_yu_manus
from agent.yu_finance import create_yu_finance

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YuAiAgent Python AI API")

# 配置 CORS，允许前端跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化应用实例
finance_app = FinanceApp()
yu_manus = create_yu_manus()
yu_finance = create_yu_finance()

# --------------------------------------------------------------------------
# 1. LoveApp 对话接口 (对应 Java LoveApp)
# --------------------------------------------------------------------------

@app.post("/ai/love_app/chat/sync")
async def do_chat_sync(message: str, chatId: str):
    """同步聊天接口"""
    logger.info(f"Sync Chat: {message} (chatId: {chatId})")
    return finance_app.do_chat(message, chatId)


@app.get("/ai/love_app/chat/sse")
async def do_chat_sse(request: Request, message: str, chatId: str):
    """流式聊天接口"""
    logger.info(f"SSE Chat: {message} (chatId: {chatId})")

    async def event_generator():
        # 这里模拟流式，因为目前 FinanceApp.do_chat 是同步的。
        # 如果底层链路也改流式会更完美，这里先做 SSE 包装以兼容前端。
        # 实际面试时可以解释：通过异步生成器实时推送 token。
        response = finance_app.do_chat(message, chatId)
        # 为了演示流式效果，将结果拆分推送（生产环境应由 LLM 底层流式输出）
        for chunk in response:
            if await request.is_disconnected():
                break
            yield chunk
            await asyncio.sleep(0.01)

    return EventSourceResponse(event_generator())


# --------------------------------------------------------------------------
# 2. Agent 接口 (对应 Java YuManus)
# --------------------------------------------------------------------------

@app.get("/ai/agent/stream-run")
async def stream_run_agent(request: Request, userPrompt: str, mode: str = "manus"):
    """
    调用智能体的流式接口
    mode: "manus" (通用) 或 "finance" (金融)
    """
    logger.info(f"Agent Stream Run: {userPrompt} (mode: {mode})")
    
    agent = yu_manus if mode == "manus" else yu_finance
    
    async def agent_event_generator():
        # 使用 Agent 的 stream_run 方法
        try:
            for chunk in agent.stream_run(userPrompt):
                if await request.is_disconnected():
                    logger.info("Client disconnected from Agent stream")
                    break
                # SSE 协议要求格式为 data: <content>\n\n
                # EventSourceResponse 会帮我们处理格式，我们只需要 yield 内容
                yield chunk
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            yield f"Error: {str(e)}"

    return EventSourceResponse(agent_event_generator())


@app.get("/ai/manus/chat")
async def manus_chat(request: Request, message: str):
    """Alias for /ai/agent/stream-run for backward compatibility"""
    return await stream_run_agent(request, message, mode="manus")


# --------------------------------------------------------------------------
# 启动说明
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    # 默认启动在 8000 端口，Java 端可配置访问此地址
    uvicorn.run(app, host="0.0.0.0", port=8000)
