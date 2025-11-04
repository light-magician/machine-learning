# coding: utf-8
get_ipython().run_line_magic('pip', 'install markdownify duckduckgo-search smolagents')
from higgingface_hub import login
get_ipython().run_line_magic('pip', 'install huggingface_hub')
get_ipython().run_line_magic('clear', '')
# set model
model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
'''
will use DuckDuckGoSearchTool for search
will build a new VisitWebpageTool to actually search the pages
'''
import re
import requests
from markdownify import markdownify
from requests.exceptions import RequestException
from smolagents import tool
@tool
def visit_webpage(url: str) -> str:
    """
    visits a webpage and returns content as markdown str
    args: url
    returns: content of webpage in markdown or error
    """
    try:
        # send a get req to url
        response = requests.get(url)
        response.raise_for_status() # except for bad status
        # convert html to markdown
        markdown_content = markdownify(response.text).strip()
        # remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        return markdown_content
    except RequestException as e:
        return f"error fetching webpage: {str(e)}}
get_ipython().run_line_magic('clear', '')
@tool
def visit_webpage(url: str) -> str:
    """
    visits a webpage and returns content as markdown str
    args: url
    returns: content of webpage in markdown or error
    """
    try:
        # send a get req to url
        response = requests.get(url)
        response.raise_for_status() # except for bad status
        # convert html to markdown
        markdown_content = markdownify(response.text).strip()
        # remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        return markdown_content
    except RequestException as e:
        return f"error fetching webpage: {str(e)}"
    except Exception as e:
        return f"an unexpected error occurred: {str(e)}"
        
get_ipython().run_line_magic('clear', '')
@tool
def visit_webpage(url: str) -> str:
    """
    visits a webpage and returns content as markdown str
    
    Args:
         url: the url to the webpage to visit
    Returns:
         content of webpage in markdown or error if request failure
    """
    try:
        # send a get req to url
        response = requests.get(url)
        response.raise_for_status() # except for bad status
        # convert html to markdown
        markdown_content = markdownify(response.text).strip()
        # remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)
        return markdown_content
    except RequestException as e:
        return f"error fetching webpage: {str(e)}"
    except Exception as e:
        return f"an unexpected error occurred: {str(e)}"
        
print(visit_webpage("https://en.wikipedia.org/wiki/Hugging_Face")[:500]
)
# create the agent
get_ipython().run_line_magic('clear', '')
from smolagents import (
CodeAgent,
ToolCallingAgent,
HfAPiModel,
DuckDuckGoSearchTool,
LiteLLMModel,
)
from smolagents import (
CodeAgent,
ToolCallingAgent,
HfApiModel,
DuckDuckGoSearchTool,
LiteLLMModel,
)
model = HfApiModel(model_id=model_id)
web_agent = ToolCallingAgent(
tools=[DuckDuckGoSearchTool(), visit_webpage],
model=model,
max_steps=10,
name="web_search_agent",
description="Runs web searches for you"
)
# 👆 that is the tool calling agent, and now we create a manager agent
get_ipython().run_line_magic('clear', '')
manager_agent = CodeAgent(
tools=[],
model=model,
managed_agents=[web_agent],
additional_authoried_imports=["time", "numpy", "pandas"],
)
get_ipython().run_line_magic('pip', 'install numpy pandas')
manager_agent = CodeAgent(
tools=[],
model=model,
managed_agents=[web_agent],
additional_authoried_imports=["time", "numpy", "pandas"],
)
manager_agent = CodeAgent(
tools=[],
model=model,
managed_agents=[web_agent],
additional_authorized_imports=["time", "numpy", "pandas"],
)
get_ipython().run_line_magic('save', 'test_manager_agents.py')
answer = manager_agent.run("If LLM training continues to scale up at the current rhythm until 2030, what would be the electric power in GW required to power the biggest training runs by 2030? What would that correspond to, compared to some countries? Please provide a source for any numbers used.")
answer = manager_agent.run("If LLM training continues to scale up at the current rhythm until 2030, what would be the electric power in GW required to power the biggest training runs by 2030? What would that correspond to, compared to some countries? Please provide a source for any numbers used.")
# ok these things do way too much work, clearly this is like a deep research type response
