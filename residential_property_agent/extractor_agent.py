import os
import logging
from collections.abc import AsyncGenerator
import tempfile

# Import Google ADK components
from google.adk.agents import (
    BaseAgent,
    InvocationContext,
)
from google.adk.events import Event
from google.adk.agents.callback_context import CallbackContext
from google.genai import types
from typing import Optional, ClassVar
from .pdf_extractor import PDFProcessor
from attachments.dspy import Attachments

# Configure logging
logger = logging.getLogger(__name__)

async def before_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    logger.info(f"Before agent callback for {callback_context.agent_name}")
    pdf_artifact = await callback_context.load_artifact("agreement.pdf")
    if pdf_artifact:
        logger.info("PDF file found in artifacts, attaching to user content.")
        callback_context.user_content.parts.append(pdf_artifact)
    return None

class ExtractorAgent(BaseAgent):

    extractor: ClassVar = PDFProcessor()

    def __init__(self, name: str):
        super().__init__(name=name, before_agent_callback=before_agent_callback)
        logger.info(f"Extractor agent '{name}' initialized.")

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        logger.info(f"Received user message parts: {len(ctx.user_content.parts)}")
        user_text = None
        pdf_attachment = None
        temp_file_path = None
        
        try:
            for part in ctx.user_content.parts:
                if part.inline_data and part.inline_data.mime_type == "application/pdf":
                    logger.info("Processing inline PDF data.")
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as f:
                        f.write(part.inline_data.data)
                        temp_file_path = f.name
                    pdf_attachment = Attachments(temp_file_path)
                elif part.text:
                    user_text = part.text
            
            prediction = self.extractor(pdf=pdf_attachment, question=user_text)
            response_text = str(prediction.answer)
            logger.info(f"Extractor agent response: {response_text}")

            yield Event(
                author=self.name,
                content=types.Content(parts=[types.Part(text=response_text)]),
                turn_complete=True,
                partial=False
            )
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    logger.error(f"Failed to delete temp file {temp_file_path}: {e}")
