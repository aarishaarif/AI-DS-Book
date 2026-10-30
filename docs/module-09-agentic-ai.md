---
title: "Module 9: Agentic AI"
sidebar_position: 9
---

# Module 9: Agentic AI

**Level:** Advanced | **Prerequisites:** Module 8 (Generative AI and LLMs)

> Agentic AI systems go beyond passive question-answering. An agent perceives its environment, plans actions, uses tools, and iterates toward a goal — autonomously and across multiple steps.

---

## Learning Objectives

- Understand the architecture of AI agents: perception, reasoning, action
- Implement tool-use (function calling) with LLMs
- Build a ReAct-style reasoning-action loop
- Design multi-agent systems with specialized roles
- Evaluate and safeguard agentic systems
- Understand key frameworks: LangChain Agents, LlamaIndex, AutoGen

## The Agent Loop

An agent combines a model with instructions, tools, state, and a stopping policy. It observes the task, decides whether it needs a tool, validates the result, updates state, and either continues or returns an answer. The loop is useful when a task requires dependent actions.

| Term | Meaning |
| --- | --- |
| Tool | A bounded operation an agent can request |
| State | Information carried between steps |
| Planning | Choosing actions toward a goal |
| Guardrail | A rule or check that limits unsafe behavior |
| Human approval | A deliberate pause before a sensitive action |

Every tool should have a narrow schema, clear failure responses, and appropriate permissions. Treat tool output as data to validate, not as instructions that can rewrite the agent's rules.

---

## 9.1 What Is an AI Agent?

A traditional LLM call is a single step:
```
Input → LLM → Output
```

An agent is a loop:
```
Goal
  → Observe environment
    → Plan (think)
      → Act (tool use / API calls)
        → Observe result
          → Revise plan
            → Act again...
              → Return final answer
```

### The Agent Components

| Component | Description |
|---|---|
| **LLM** | The "brain" — reasons and decides |
| **Tools** | Functions the agent can call (search, calculator, code runner) |
| **Memory** | Short-term (conversation history) and long-term (vector store) |
| **Planning** | Strategy for multi-step task completion |
| **Environment** | Where the agent operates (web, file system, APIs) |

---

## 9.2 Tool Use (Function Calling)

Function calling allows the LLM to output structured tool invocations instead of plain text.

### Google Gemini Function Calling

```python
import google.generativeai as genai
import os, json, math

genai.configure(api_key=os.environ['GEMINI_API_KEY'])

# Define tools the agent can use
tools = [
    {
        "function_declarations": [
            {
                "name": "calculate",
                "description": "Perform a mathematical calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "A mathematical expression to evaluate, e.g. '2 + 2'",
                        }
                    },
                    "required": ["expression"],
                },
            },
            {
                "name": "search_web",
                "description": "Search the web for current information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        }
                    },
                    "required": ["query"],
                },
            },
        ]
    }
]

# Tool implementations
def calculate(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, {"sqrt": math.sqrt, "pi": math.pi})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def search_web(query: str) -> str:
    # Mock implementation — replace with real search API
    return f"[Mock search results for: '{query}']"

TOOL_MAP = {"calculate": calculate, "search_web": search_web}

# Agentic loop
model = genai.GenerativeModel('gemini-2.5-flash', tools=tools)
chat = model.start_chat()

response = chat.send_message("What is the square root of 144, and then multiply it by pi?")

# Process tool calls
while response.candidates[0].content.parts[0].function_call.name:
    fc = response.candidates[0].content.parts[0].function_call
    tool_name = fc.name
    tool_args = dict(fc.args)

    print(f"Agent calling: {tool_name}({tool_args})")
    tool_result = TOOL_MAP[tool_name](**tool_args)
    print(f"Tool result: {tool_result}")

    # Send result back to model
    response = chat.send_message(
        genai.protos.Content(parts=[
            genai.protos.Part(
                function_response=genai.protos.FunctionResponse(
                    name=tool_name,
                    response={"result": tool_result},
                )
            )
        ])
    )

print(f"\nFinal answer: {response.text}")
```

---

## 9.3 ReAct: Reasoning + Acting

The ReAct framework interleaves **reasoning** (thought) and **action** (tool calls):

```
Thought: I need to find the current price of NVDA stock.
Action: search_web("NVDA stock price today")
Observation: NVDA is trading at $875.

Thought: Now I need to calculate 10% of $875.
Action: calculate("875 * 0.10")
Observation: 87.5

Thought: I have all the information needed.
Final Answer: 10% of NVDA's current price is $87.50.
```

### Implementing ReAct from Scratch

```python
import google.generativeai as genai
import re

REACT_SYSTEM_PROMPT = """You are an AI agent that solves tasks step by step.
You have access to these tools:
- calculate(expression): Evaluate a math expression
- search_web(query): Search for current information

Format your response as:
Thought: <your reasoning>
Action: <tool_name>(<argument>)

After seeing the observation, continue reasoning until you reach:
Final Answer: <your final answer>

Never skip the Final Answer step."""

def run_react_agent(question: str, max_steps: int = 10) -> str:
    model = genai.GenerativeModel(
        'gemini-2.5-flash',
        system_instruction=REACT_SYSTEM_PROMPT,
    )
    chat = model.start_chat()
    response = chat.send_message(question)

    for step in range(max_steps):
        text = response.text
        print(f"\n--- Step {step+1} ---\n{text}")

        # Check for Final Answer
        if "Final Answer:" in text:
            return text.split("Final Answer:")[-1].strip()

        # Parse action
        action_match = re.search(r'Action:\s*(\w+)\((.+?)\)', text)
        if not action_match:
            return text

        tool_name = action_match.group(1)
        tool_arg  = action_match.group(2).strip('"\'')

        # Execute tool
        if tool_name in TOOL_MAP:
            observation = TOOL_MAP[tool_name](tool_arg)
        else:
            observation = f"Unknown tool: {tool_name}"

        print(f"Observation: {observation}")
        response = chat.send_message(f"Observation: {observation}")

    return "Max steps reached without a final answer."

result = run_react_agent("What is 25% of 360?")
print(f"\nResult: {result}")
```

---

## 9.4 Memory Systems

### Short-Term Memory (Conversation Buffer)

```python
# Conversation history acts as short-term memory
chat = model.start_chat(history=[
    {"role": "user", "parts": ["My name is Arisha."]},
    {"role": "model", "parts": ["Hello Arisha! How can I help you today?"]},
])

response = chat.send_message("What's my name?")
print(response.text)  # The model remembers: "Your name is Arisha."
```

### Long-Term Memory (Vector Store)

```python
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
import google.generativeai as genai

# Persistent memory: store and retrieve past interactions
client = QdrantClient(path="./agent_memory")
client.recreate_collection(
    collection_name="memories",
    vectors_config=qm.VectorParams(size=768, distance=qm.Distance.COSINE),
)

def store_memory(text: str, memory_id: int):
    embedding = genai.embed_content(
        model='models/gemini-embedding-001',
        content=text,
        task_type='retrieval_document',
    )['embedding']
    client.upsert("memories", [qm.PointStruct(id=memory_id, vector=embedding, payload={"text": text})])

def recall_memories(query: str, top_k: int = 3) -> list:
    q_emb = genai.embed_content(
        model='models/gemini-embedding-001',
        content=query,
        task_type='retrieval_query',
    )['embedding']
    results = client.search("memories", query_vector=q_emb, limit=top_k)
    return [r.payload["text"] for r in results]

# Store some memories
store_memory("User prefers Python over JavaScript.", 1)
store_memory("User is learning about machine learning.", 2)
store_memory("User's project involves time-series forecasting.", 3)

# Recall relevant memories
memories = recall_memories("What programming language should I use?")
print("\n".join(memories))
```

---

## 9.5 Multi-Agent Systems

Multiple specialized agents collaborate to solve complex tasks.

### Orchestrator–Worker Pattern

```python
class Agent:
    def __init__(self, name: str, role: str, tools: list = None):
        self.name = name
        self.role = role
        self.model = genai.GenerativeModel(
            'gemini-2.5-flash',
            system_instruction=f"You are {name}. {role}",
        )

    def run(self, task: str) -> str:
        response = self.model.generate_content(task)
        return response.text


# Define specialized agents
researcher = Agent(
    name="Researcher",
    role="You search for and summarize factual information. Be concise.",
)

writer = Agent(
    name="Writer",
    role="You write clear, engaging explanations based on provided research.",
)

reviewer = Agent(
    name="Reviewer",
    role="You review content for accuracy and clarity. Point out any issues.",
)


def multi_agent_pipeline(topic: str) -> str:
    print(f"\n=== Multi-Agent Pipeline: {topic} ===\n")

    # Step 1: Research
    research = researcher.run(f"Research key facts about: {topic}")
    print(f"[Researcher]:\n{research}\n")

    # Step 2: Write
    draft = writer.run(f"Write a short explanation about {topic} based on:\n{research}")
    print(f"[Writer]:\n{draft}\n")

    # Step 3: Review
    review = reviewer.run(f"Review this explanation about {topic}:\n{draft}")
    print(f"[Reviewer]:\n{review}\n")

    return draft


result = multi_agent_pipeline("gradient descent in neural networks")
```

---

## 9.6 Agent Safety and Guardrails

Agentic systems need safeguards to prevent unintended actions.

```python
DANGEROUS_PATTERNS = [
    "rm -rf", "delete all", "drop table", "format disk",
    "shutdown", "sudo", "exec(",
]

def safe_execute_tool(tool_name: str, tool_input: str) -> str:
    """Validate tool inputs before execution."""
    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if pattern.lower() in tool_input.lower():
            return f"BLOCKED: Input '{tool_input}' contains dangerous pattern '{pattern}'"

    # Rate limiting check (simplified)
    if len(tool_input) > 1000:
        return "BLOCKED: Input exceeds maximum length"

    return TOOL_MAP.get(tool_name, lambda x: "Tool not found")(tool_input)


class SafeAgent:
    def __init__(self, allowed_tools: list, max_steps: int = 10):
        self.allowed_tools = allowed_tools
        self.max_steps = max_steps
        self.action_log = []

    def validate_action(self, tool_name: str) -> bool:
        return tool_name in self.allowed_tools

    def log_action(self, tool: str, input_: str, output: str):
        self.action_log.append({"tool": tool, "input": input_, "output": output})
```

---

## 9.7 Key Agent Frameworks

| Framework | Strengths | Best For |
|---|---|---|
| **LangChain** | Rich tool ecosystem, chains, agents | General-purpose agents |
| **LlamaIndex** | Document indexing, RAG-focused | Knowledge retrieval agents |
| **AutoGen (Microsoft)** | Multi-agent conversations | Collaborative agents |
| **CrewAI** | Role-based multi-agent | Structured pipelines |
| **Google ADK** | Gemini-native agents | Google ecosystem |

### LangChain Agent (Quick Example)

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

tools = [
    Tool(
        name="Calculator",
        func=lambda x: str(eval(x)),
        description="Evaluates math expressions",
    )
]

agent = initialize_agent(
    tools, llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

agent.run("What is 456 * 789?")
```

---

## Common Mistakes

- Building agents without a budget/step limit — they can run indefinitely
- Not logging agent actions — impossible to debug without an audit trail
- Giving agents irreversible tools (file deletion, emails) without confirmation
- Over-engineering a simple prompt-response pattern as an agent
- Trusting agent-generated tool inputs without validation

---

## Practice Exercises

1. Build a calculator agent with Gemini function calling that handles multi-step math problems.
2. Implement a ReAct agent that can look up Python documentation and answer coding questions.
3. Create a two-agent pipeline: a "planner" that breaks a task into steps, and an "executor" that completes them.
4. Add a long-term memory system to an agent that remembers user preferences across sessions.
5. Implement input validation for an agent's tools that blocks dangerous shell commands.

---

## Review Questions

1. How does an agent's observe, decide, act, and evaluate loop differ from a single model call?
2. What information should a tool schema validate before execution?
3. How does ReAct use observations to guide a later action?
4. What is the difference between short-term conversational context and persistent memory?
5. Which permissions and limits reduce risk when an agent uses tools?

## Further Reading

- [ReAct: Synergizing Reasoning and Acting (paper)](https://arxiv.org/abs/2210.03629)
- [LangChain Agents Documentation](https://python.langchain.com/docs/modules/agents/)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [Google AI Agent Development Kit](https://google.github.io/adk-docs/)
- [Voyager: An Open-Ended Embodied Agent (paper)](https://arxiv.org/abs/2305.16291)
