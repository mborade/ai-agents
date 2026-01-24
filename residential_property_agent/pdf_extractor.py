import os
import argparse
import dspy
import json
import logging
from dotenv import load_dotenv
from attachments.dspy import Attachments 
from pydantic import BaseModel, Field
from typing import List

# Configure logging
logger = logging.getLogger(__name__)

lm = dspy.LM("vertex_ai/gemini-2.5-pro", temperature=0.1)
dspy.configure(lm=lm)

class ContactInfo(BaseModel):
    """Important contact information"""

    seller_name: str = Field(description="Seller's name")
    buyer_name: str = Field(description="Buyer's name")
    seller_agent_name: str = Field(description="Seller's agent's name")
    buyer_agent_name: str = Field(description="Buyer's agent's name")


class PDFExtractInfo(BaseModel):
    """
    Extract contact, timeline and terms related information from a PDF file.
    """
    contact_info: ContactInfo = Field(description="Contact information extracted from the PDF file.")
    timeline_info: List[str] = Field(description="Timeline information extracted from the PDF file.")
    terms: List[str] = Field(description="Terms extracted from the PDF file.")


class PDFExtractOutput(dspy.Signature):
    """
    Extract data from a PDF file. Provide correct answer to the question based on the PDF file and citing the section of the PDF file used to answer the question.
    """
    pdf: Attachments = dspy.InputField(desc="The PDF file to extract data from.")
    question: str = dspy.InputField(desc="The question to answer based on the PDF file.")
    # extracted_data: PDFExtractInfo = dspy.OutputField(desc="Extracted data from the PDF file.")
    answer: str = dspy.OutputField(desc="Answer to the question based on the PDF file.")

# class BasicQA(dspy.Signature):
#     """Answer questions based on general knowledge."""
#     question: str = dspy.InputField()
#     answer: str = dspy.OutputField()

class PDFProcessor(dspy.Module):
    """
    Process a residential purchase agreement PDF file and extract data from it.
    """
    def __init__(self):
        super().__init__()
        self.pdf_extractor = dspy.ChainOfThought(signature=PDFExtractOutput)
        # self.query_handler = dspy.ChainOfThought(signature=BasicQA)
        
    def forward(self, pdf: Attachments = None, question: str = None):
        # if pdf:
        logger.info(f"extract pdf {pdf}")
        return self.pdf_extractor(pdf=pdf, question=question)
        # else:
        #     return self.query_handler(question=question)


def main():
    parser = argparse.ArgumentParser(description="PDF Extractor")
    parser.add_argument("pdf_path", help="Path to PDF file")
    args = parser.parse_args()
    
    # Ensure logging is configured for CLI usage
    logging.basicConfig(level=logging.INFO)
    
    pdf_processor = PDFProcessor()
    pdf = Attachments(args.pdf_path)
    result = pdf_processor(pdf=pdf)
    dspy.inspect_history(n=1)
    logger.info(result)

if __name__ == "__main__":
    main()
