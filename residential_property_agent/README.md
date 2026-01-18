# Residential Property Agent

This module implements an AI agent capable of parsing residential purchase agreement PDFs to extract critical information such as contact details, timelines, and terms. It leverages **DSPy** for robust prompting and **Google's Agent Development Kit (ADK)** for agent orchestration.

## Overview

The Residential Property Agent is designed to automate the extraction of structured data from complex legal documents. It consists of a multi-agent system where an orchestrator delegates PDF processing tasks to a specialized extractor agent.

### Key Features
*   **Automated Information Extraction**: Identifies and extracts names of buyers, sellers, and their respective agents.
*   **Timeline & Terms Analysis**: Parses the document to find key dates and contractual terms.
*   **PDF Processing**: Directly handles PDF attachments to process text content.
*   **DSPy Integration**: Uses `dspy.ChainOfThought` and custom Signatures to improve extraction accuracy and reasoning.
*   **Vertex AI Powered**: Built on top of Google's Gemini models (via Vertex AI).

## Architecture

The solution is organized into three main components:

1.  **`orchestrator_agent.py`**:
    *   Acts as the main interface for the user.
    *   Handles file uploads (saving PDF artifacts).
    *   Delegates specific questions about the agreement to the `extractor_agent`.

2.  **`extractor_agent.py`**:
    *   A custom agent (`BaseAgent`) that receives the PDF and user queries.
    *   Wraps the DSPy processing logic.
    *   Manages the conversion of PDF content into a format suitable for the LLM.

3.  **`pdf_extractor.py`**:
    *   Defines the data structures (`pydantic` models) for the extracted info (ContactInfo, PDFExtractInfo).
    *   Defines the DSPy `Signature` and `Module` (`PDFProcessor`) used for the reasoning logic.

## Prerequisites

*   Python 3.10+
*   Google Cloud Project with Vertex AI API enabled.
*   `google-adk`
*   `dspy-ai`
*   `pydantic`

## Setup

1.  **Environment Variables**:
    Ensure the following environment variables are set (typically in your `.env` file or runtime environment):
    ```bash
    GOOGLE_CLOUD_PROJECT=<your-project-id>
    GOOGLE_CLOUD_LOCATION=us-central1 # or your preferred region
    GOOGLE_GENAI_USE_VERTEXAI=True
    ```

2.  **Dependencies**:
    Install the required Python packages:
    ```bash
    pip install dspy-ai google-adk pydantic
    ```

## Usage

The agent is typically run via the Agent Engine or a local runner.

**Example usage flow:**
1.  User sends a message with a PDF attachment of a residential purchase agreement.
2.  `orchestrator_agent` receives the message, saves the PDF as an artifact, and recognizes the need for extraction.
3.  It calls the `extractor_agent`.
4.  `extractor_agent` reads the PDF artifact, passes the content to the DSPy `PDFProcessor`, and returns the extracted answers or structured data.

## Code Structure

```python
# pdf_extractor.py - Data Models
class ContactInfo(BaseModel):
    seller_name: str
    buyer_name: str
    ...

# pdf_extractor.py - DSPy Module
class PDFProcessor(dspy.Module):
    def forward(self, pdf, question):
        # Chain of thought execution
        ...
```
