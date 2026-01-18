import os

import google.auth
from google.adk.agents import Agent
from google.adk.apps.app import App
from google.adk.tools import agent_tool
from google.adk.agents.callback_context import CallbackContext
from google.genai import types
from typing import Optional
from .extractor_agent import ExtractorAgent

_, project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")


async def before_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    print(f"Before agent callback for {callback_context.agent_name}, user content parts: {callback_context.user_content.parts}")
    for part in callback_context.user_content.parts:
        if part.inline_data and part.inline_data.mime_type == "application/pdf":
            print("PDF file uploaded by user.")
            pdf_bytes = part.inline_data.data
            agreement_artifact = types.Part.from_bytes(
                data=pdf_bytes,
                mime_type="application/pdf"
            )
            file_name = 'agreement.pdf'
            await callback_context.save_artifact(filename=file_name, artifact=agreement_artifact)
        # filename = part.inline_data.display_name
        # callback_context.session.state["pdf_bytes"] = pdf_bytes
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