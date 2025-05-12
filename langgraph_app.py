from pydantic import BaseModel, field_validator, ValidationError
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_deepseek import ChatDeepSeek
import re
from langgraph.errors import GraphRecursionError
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import WikipediaLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import ClassVar, Any
from pydantic import BaseModel, Field
import requests
import httpx
import base64

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
class SearchInput(BaseModel):
    query: str = Field(description="should be a search query")

class MyRetreiver(BaseTool):
    load_dotenv()
    name: ClassVar[str] = "Wikipedia_Search_Tool"
    description: ClassVar[str] = "useful for when you need to search for information in Wikipedia"
    child_splitter: ClassVar[Any] = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    docstore: ClassVar[Any] = InMemoryStore()
    vectorstore: ClassVar[Any] = Chroma(embedding_function=GoogleGenerativeAIEmbeddings(model='models/embedding-001'), collection_name='wikipedia_docs', persist_directory='./chroma_db')
    retriever: ClassVar[Any] = ParentDocumentRetriever(vectorstore=vectorstore, docstore=docstore, child_splitter=child_splitter)
    def __init__(self):
        super().__init__()
        docs = WikipediaLoader('*', load_max_docs=2000, doc_content_chars_max=10000).load()
        
        self.args_schema = SearchInput
        self.retriever.add_documents(docs, ids=None)

    def _run(
        self, query: str
    ) -> str:
        """Use the tool."""
        retrieved_docs = self.retriever.invoke(query)
        print(f"Retrieved docs: {len(retrieved_docs)}")
        if len(retrieved_docs) == 0:
            return "No relevant documents found"
        else:
            return retrieved_docs[0].page_content
        
class MyWikipediaLoader(BaseTool):
    load_dotenv()
    name: ClassVar[str] = "wikipedia_loader_tool"
    description: ClassVar[str] = "useful for when you need to search for information in Wikipedia"
    return_direct: ClassVar[bool] = False
    def __init__(self):
        super().__init__()
        self.args_schema = SearchInput
        # self.return_direct = False

    def _run(
        self, query: str
    ) -> str:
        """Use the tool."""
        retrieved_docs = WikipediaLoader(query, load_max_docs=6, doc_content_chars_max=10000).load()
        print(f"Retrieved docs: {len(retrieved_docs)}")
        if len(retrieved_docs) == 0:
            return "No relevant documents found"
        else:
            return "\n\n***************************************************************************".join([doc.page_content for doc in retrieved_docs])
            
    
class CustomMessagesState(MessagesState):
    task_id: str

class MyAgent:
    def __init__(self, llm=None):
        load_dotenv()
        if not llm:
            # llm = ChatOpenAI(model="gpt-4.1", temperature=0, max_retries=2)
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-preview-05-06", temperature=0, max_tokens=None, timeout=None, max_retries=15) 
            #gemini-2.0-flash, gemini-2.5-pro-preview-05-06
            
            # llm = ChatDeepSeek(
            #     model="deepseek-chat",
            #     temperature=0,
            #     max_tokens=None,
            #     timeout=None,
            #     max_retries=2,
            #     # other params...
            # )
        tools = [TavilySearchResults(max_results=3), 
                #  MyWikipediaLoader(),
                  WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=6, doc_content_chars_max=10000), response_format='content')
                ]
        self.tools = tools
        self.llm = llm.bind_tools(tools)
        # System message
        self.sys_msg = SystemMessage(content="You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].  YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string. Try to use Wikipedia first to get the required information. Skip the questions which requires referring to an attached file, video or audio.")

        # Graph
        builder = StateGraph(CustomMessagesState)

        # Define nodes: these do the work
        builder.add_node("assistant", self.assistant)
        builder.add_node("process_file", self.process_file)
        builder.add_node("process_audio", self.process_audio)
        builder.add_node("process_image", self.process_image)
        builder.add_node("tools", ToolNode(tools))

        # Define edges: these determine how the control flow moves
        builder.add_conditional_edges(START, self.ismultimodal)
        builder.add_edge("process_file", "assistant")
        builder.add_edge("process_audio", "assistant")
        builder.add_edge("process_image", "assistant")
        builder.add_conditional_edges(
            "assistant",
            # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
            # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
            tools_condition,
        )
        builder.add_edge("tools", "assistant")
        self.react_graph = builder.compile() 

        from IPython.display import Image, display
        from langchain_core.runnables.graph import MermaidDrawMethod

        # display(Image(self.react_graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.PYPPETEER)))

    def __call__(self, task_id:str, question: str) -> str:
            print(f"Agent received question (first 50 chars): {question[:50]}...")
            fixed_answer = self.run(task_id, question)
            # fixed_answer = None
            print(f"Agent returning fixed answer: {fixed_answer}")
            return fixed_answer
    
    def assistant(self, state: CustomMessagesState):
        return {"messages": self.llm.invoke([self.sys_msg] + state["messages"])}

    def process_audio(self, state: CustomMessagesState):
        print(f"processing audio file")
        audio_url = f"{DEFAULT_API_URL}/files/{state['task_id']}"
        audio_data = base64.b64encode(httpx.get(audio_url).content).decode("utf-8")
        state["messages"] = [SystemMessage(content="You are an AI assistant who is asked to transcribe an audio file. You should respond with only the transcribed text"),HumanMessage(content=[{
            "type": "text",
            "text": "Transcribe this audio:",
        },
        {
            "type": "input_audio",
            "input_audio": {"data": audio_data, "format": "mp3"},
            # "source_type": "base64",
            # "data": audio_data,
            # "mime_type": "audio/mp3"
        }])]
        return state

    def process_video(self, state: CustomMessagesState):
        print(f"processing video file")
        return
    
    def process_image(self, state: CustomMessagesState):
        print(f"processing image file")
        image_url = f"{DEFAULT_API_URL}/files/{state['task_id']}"
        image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
        state["messages"] = [HumanMessage(content=[{
            "type": "text",
            "text": "Analyze the image",
        },
        {
            "type": "image_url",
            "image_url": {
            "url":  f"data:image/png;base64,{image_data}"
          }}])]
        return state
    
    def process_file(self, state: CustomMessagesState):
        print(f"processing file")
        response = requests.get(f"{DEFAULT_API_URL}/files/{state['task_id']}")
        response.raise_for_status()
        file_content = response.text
        print(f"attached file content {file_content}")
        state['messages'] = HumanMessage(content=[{"type": "text", "text": "Refer to the file content"}, {"type": "text", "text": file_content}])
        return state

    def ismultimodal(self, state: CustomMessagesState):
        print(f"Messages: {state['messages'][0].content}")
        # if any(value in state["messages"][0].content for value in ["youtube", "attached", "image"]):    
        if 'youtube' in state['messages'][0].content:
            print("received video file")    
            return 'process_video'
        elif any(s in state['messages'][0].content for s in ['.mp3', 'audio']):
            print("received attached file")    
            return 'process_audio'
        elif any(s in state['messages'][0].content for s in['.jpg', 'image']):
            print("received image file")    
            return 'process_image'
        elif 'attached' in state["messages"][0].content:
            return 'process_file'
        else:
            return "assistant"
            


    def run(self, task_id:str,question: str) -> str:
        try:
            result = self.react_graph.invoke({"messages": [HumanMessage(content=question)], "task_id": task_id}, {"max_recursion": 10})
            print(f"messages: {result}")
            # for m in result['messages']:
            #     if m.type == "ai":
                    # print(f"ai message {m.content}")
            print(f"final message {result['messages'][-1].content}")
            if "FINAL ANSWER:" not in result['messages'][-1].content:
                return result['messages'][-1].content
            result = re.split(r'FINAL ANSWER: ', result['messages'][-1].content, 1)[1]
            print(f"result {result}")
            return result
            # result = self.llm.invoke([self.sys_msg] + [question])
            # print(result)
            # # print(result.tool_calls)
        except GraphRecursionError:
            print("Recursion Error")

if __name__ == "__main__":
    agent = MyAgent()
    print(agent("cca530fc-4052-43b2-b130-b30968d8aa44", """Review the chess position provided in the image. It is black's turn. Provide the correct next move for black which guarantees a win. Please provide your response in algebraic notation."""))