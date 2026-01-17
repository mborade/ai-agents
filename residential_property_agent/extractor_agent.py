import os
from collections.abc import AsyncGenerator
from datetime import datetime
from enum import Enum
from typing import ClassVar

import io
import dspy
import google.auth

# Import Google ADK components
from google.adk.agents import (
    BaseAgent,
    InvocationContext,
)
from google.adk.apps.app import App
from google.adk.events import Event
from google.adk.agents.callback_context import CallbackContext
from google.genai import types
from typing import Optional
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel
from .pdf_extractor import PDFProcessor
from attachments.dspy import Attachments
import tempfile

_, project_id = google.auth.default()
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

async def before_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    print(f"Before agent callback for {callback_context.agent_name}, user content parts: {callback_context.user_content.parts}")
    pdf_artifact = await callback_context.load_artifact("agreement.pdf")
    if pdf_artifact:
        print("PDF file found in user content.")
        callback_context.user_content.parts.append(pdf_artifact)
    return None

class ExtractorAgent(BaseAgent):

    extractor: ClassVar = PDFProcessor()

    def __init__(self, name:str):
        super().__init__(name=name, before_agent_callback=before_agent_callback)
        # Initialize the DSPy module once during agent startup
        print("Extractor agent initialized.")

    # The 'run' method is the entry point for the Agent Engine  # noqa: F821
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # 1. Extract User Input
        # The context contains the user's message and session state
        print(f"Received user message: {ctx.user_content.parts}")
        user_text = None
        pdf_attachment = None
        
        # Iterating through all the parts in user content to find the PDF file
        print(f"session events: {ctx.session.events}")
        for part in ctx.user_content.parts:
            if part.inline_data and part.inline_data.mime_type == "application/pdf":
                print("PDF file found in user content.")
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as f:
                    f.write(part.inline_data.data)
                    print(f.name)
                pdf_attachment = Attachments(f.name)
                print(f"extracted pdf attachment: {pdf_attachment}")
            if part.text:
                user_text = part.text
        
        prediction = self.extractor(pdf=pdf_attachment, question=user_text)

        # 3. Process DSPy Output
        # Access the fields defined in your Signature
        # intent = prediction.intent
        # response_text = str(prediction.extracted_data) + "\n\n" + 
        response_text = str(prediction.answer)
        # rationale = prediction. # Captured automatically by ChainOfThought

        # Optional: Log the reasoning trace
        print(f"Extractor agent response: {response_text}")

        # 4. Yield Response back to Vertex AI Agent platform
        yield Event(author="mangb",
                    content=types.Content(parts=[types.Part(
                    text=str(response_text)
                )]),
                turn_complete=True,
                partial=False
                )


