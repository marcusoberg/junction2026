import asyncio
import os
import sys
import cv2
import numpy as np

from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types


def make_test_image_bytes(color_name: str = "blue") -> bytes:
    img = np.full((240, 320, 3), 230, dtype=np.uint8)
    bgr_map = {"red": (0, 0, 220), "yellow": (0, 220, 220), "blue": (220, 0, 0)}
    bgr = bgr_map.get(color_name, (0, 200, 0))
    cv2.rectangle(img, (110, 80), (210, 160), bgr, -1)
    cv2.putText(img, "WRIST_CAM", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def inspect_test_camera(view: str = "wrist") -> dict:
    """Capture camera image and metadata."""
    data = make_test_image_bytes("blue" if view == "wrist" else "red")
    return {
        "view": view,
        "note": "Square block centered in view.",
        "image": types.Part.from_bytes(data=data, mime_type="image/jpeg"),
    }


async def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("ADK_MODEL", "gemini-2.5-flash")
    print(f"Testing ADK with Gemini model={model_name!r}, GEMINI_API_KEY={bool(os.getenv('GEMINI_API_KEY'))}, GOOGLE_API_KEY={bool(os.getenv('GOOGLE_API_KEY'))}")
    agent = LlmAgent(
        name="vision_smoke",
        model=Gemini(model=model_name),
        instruction="Call inspect_test_camera with view='wrist' and report the exact color of the rectangle you see in the image.",
        tools=[FunctionTool(inspect_test_camera)],
    )
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name="smoke", user_id="u1")
    runner = Runner(agent=agent, app_name="smoke", session_service=session_service)
    content = types.Content(role="user", parts=[types.Part.from_text(text="Check wrist camera and tell me what color block is in the image.")])
    async for event in runner.run_async(user_id="u1", session_id=session.id, new_message=content):
        if event.content and event.content.parts:
            for p in event.content.parts:
                if p.text:
                    print(f"[{event.author}] {p.text}")
                elif p.function_call:
                    print(f"[{event.author}] CALL {p.function_call.name}({p.function_call.args})")
                elif p.function_response:
                    print(f"[{event.author}] RESP {p.function_response.name}: keys={list((p.function_response.response or {}).keys())}, parts={len(p.function_response.parts or [])}")


if __name__ == "__main__":
    asyncio.run(main())
