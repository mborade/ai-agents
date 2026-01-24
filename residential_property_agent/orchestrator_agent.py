import os
import logging

import google.auth
from google.adk.agents import Agent
from google.adk.apps.app import App
from google.adk.tools import agent_tool
from google.adk.agents.callback_context import CallbackContext
from google.genai import types
from typing import Optional
from .extractor_agent import ExtractorAgent

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_, project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")


async def before_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    logger.info(f"Before agent callback for {callback_context.agent_name}")
    for part in callback_context.user_content.parts:
        if part.inline_data and part.inline_data.mime_type == "application/pdf":
            logger.info("PDF file identified in user message; saving as artifact.")
            pdf_bytes = part.inline_data.data
            agreement_artifact = types.Part.from_bytes(
                data=pdf_bytes,
                mime_type="application/pdf"
            )
            await callback_context.save_artifact(filename='agreement.pdf', artifact=agreement_artifact)
    return None


extractor_agent = ExtractorAgent("extractor_agent")

root_agent = Agent(
    name="orchestrator_agent",
    model="gemini-2.5-pro",
    instruction="You are a helpful AI assistant designed to provide accurate and useful information. Always delegate to extractor agent for any query related to the agreement. Also, ensure that any document uploaded should be passed to the extractor agent if not already passed when the extractor agent is invoked.",
    tools=[agent_tool.AgentTool(agent=extractor_agent)],
    before_agent_callback=before_agent_callback
)

app = App(root_agent=root_agent, name="residential_property_agent")