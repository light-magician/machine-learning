# coding: utf-8
get_ipython().run_line_magic('pip', 'install pandas langchain langchain-community sentence-transformers datasets python-dotenv rank_bm25')
get_ipython().run_line_magic('clear', '')
from dotenv import load_dotenv
load_dotenv()
import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
knowlege_bas = datasets.load_dataset("m-ric/huggingface_doc", split="train")
knowlege_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))
knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))
source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    separators=["\n\n", "\n", ".", " ", ""],
)
get_ipython().run_line_magic('clear', '')
# documents are ready
get_ipython().run_line_magic('save', 'test_smolagents.py')
get_ipython().run_line_magic('clear', '')
from smolagents import Tool
class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
        inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "the query to perform. Should be semantically close to your target documents. Use the affirmative form rather than a question."
            }
     }
     output_type = "string" 
inputs = {
    "query": {
        "type": "string",
        "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
    }
}
class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of transformers documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(
            docs, k=10
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )
        
retriever_tool = RetrieverTool(docs_processed)
docs_processed = text_splitter.split_documents(source_docs)
retriever_tool = RetrieverTool(docs_processed)
"""
using BM25 a classic retrieval method bc its lightning fast to setup
to improve retrieval accuracy, we could replace with semantic search using vector representation for documents
thus you can head to the MTEB leaderboard to select a good embedding model https://huggingface.co/spaces/mteb/leaderboard
"""
from smolagents import HfApiModel, CodeAgent
agent = CodeAgent(
tools=[retriever_tool], model=HfApiModel(), max_steps=4, verbosity_level=2
)
angent.run("for a transformer model training, which is slower the forward or backward pass")
agent.run("for a transformer model training, which is slower the forward or backward pass")
