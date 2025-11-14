# Deepagent Quickstarts 

## Overview

[Deepagents](https://github.com/langchain-ai/deepagents) is a simple, opinionated agent harness build on top of [LangGraph](https://github.com/langchain-ai/langgraph). It features a few general, built-in tools that are useful for building many type of agents. 
 
These built-in tools follow a few common patterns used across popular agents such as [Manus](https://rlancemartin.github.io/2025/10/15/manus/) and [Claude Code](https://www.claude.com/product/claude-code), including the ability to work, use a filesystem/shell, and delegate work to subagents. This repo has a collection of quickstarts that demonstrate different agents that can he easily configured using the `deepagents` package.

### Tools

Every deepagent comes with a set of general tools by default:

| Tool Name | Description |
|-----------|-------------|
| `write_todos` | Create and manage structured task lists for tracking progress through complex workflows |
| `ls` | List all files in a directory (requires absolute path) |
| `read_file` | Read content from a file with optional pagination (offset/limit parameters) |
| `write_file` | Create a new file or completely overwrite an existing file |
| `edit_file` | Perform exact string replacements in files |
| `glob` | Find files matching a pattern (e.g., `**/*.py`) |
| `grep` | Search for text patterns within files |
| `execute` | Run shell commands in a sandboxed environment (only if backend supports SandboxBackendProtocol) |
| `task` | Delegate tasks to specialized sub-agents with isolated context windows |

### Middleware

Deepagent also use some built-in ["middleware"](https://docs.langchain.com/oss/python/langchain/middleware/overview), which can:

1. **Provide tools** - Add new tools to the agent's toolkit (e.g., `FilesystemMiddleware` adds `ls`, `read_file`, `write_file`, etc.)
2. **Wrap model calls** - Inject system prompts and modify model requests before they're sent
3. **Wrap tool calls** - Process tool call results after tools execute (e.g., `SummarizationMiddleware` summarizes large conversation history)

Every deepagent includes the following middleware by default (applied in order). Some middleware are provided by the `deepagents` package (`FilesystemMiddleware`, `SubAgentMiddleware`, `PatchToolCallsMiddleware`), while others come from `langchain` (`TodoListMiddleware`, `SummarizationMiddleware`, `HumanInTheLoopMiddleware`) and `langchain-anthropic` (`AnthropicPromptCachingMiddleware`):

| Middleware | Tools Added | Where It Acts | What It Does |
|------------|-------------|---------------|--------------|
| **TodoListMiddleware** | `write_todos`, `read_todos` | `wrap_model_call`, `before_agent` | Provides task planning and progress tracking tools. Enables agents to create structured todo lists, break down complex tasks into steps, and track completion status. Injects todo usage instructions into system prompt and makes current todo state available to the agent. |
| **FilesystemMiddleware** | `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`* | `wrap_model_call`, `wrap_tool_call` | Provides file system operations and context offloading. Adds filesystem tools for reading, writing, and searching files. In `wrap_model_call`: Injects filesystem instructions into system prompt and filters out `execute` tool if backend doesn't support SandboxBackendProtocol. In `wrap_tool_call`: Intercepts large tool results (>20,000 tokens), automatically saves them to files, and returns summaries instead of full content to prevent context overflow. |
| **SubAgentMiddleware** | `task` | `wrap_model_call` | Enables task delegation to specialized subagents with isolated contexts. Provides the `task` tool for spawning ephemeral subagents that can handle complex, multi-step tasks independently. Subagents run with their own context windows and return concise summaries, preventing context pollution in the main agent. Injects detailed instructions about when and how to use subagents effectively (parallel execution, context isolation, etc.). |
| **SummarizationMiddleware** | N/A | `before_agent` | Prevents context window overflow via automatic summarization. Monitors conversation history token count before each agent turn. When tokens exceed 170,000, summarizes older messages while keeping the last 6 messages intact. Replaces summarized portion with a concise summary, maintaining conversation continuity while freeing up context space. |
| **AnthropicPromptCachingMiddleware** | N/A | `wrap_model_call` | Reduces API costs through prompt caching (Anthropic models only). Adds cache control headers to system prompts, marking static content for caching on Anthropic's servers. Configured with `unsupported_model_behavior="ignore"` to allow non-Anthropic models to work without errors. Significantly reduces costs for repeated system prompts by reusing cached prefixes. |
| **PatchToolCallsMiddleware** | N/A | `before_agent` | Fixes "dangling" tool calls from interrupted operations. Scans message history before each agent turn to find AIMessages with tool_calls that lack corresponding ToolMessages. Adds placeholder ToolMessages (e.g., "Tool call X was cancelled - another message came in before it could be completed") to prevent LangGraph validation errors. Essential for handling user interruptions and double-texting scenarios. |
| **HumanInTheLoopMiddleware** | N/A | `wrap_tool_call` | Enables human approval for sensitive operations. Intercepts tool calls for tools specified in the `interrupt_on` configuration. Creates LangGraph interrupts/breakpoints that pause execution and wait for human approval or rejection. Requires a checkpointer to maintain state across interruptions. Only included in deepagents when `interrupt_on` parameter is provided to `create_deep_agent()`. |

\* The `execute` tool is only available if the backend implements `SandboxBackendProtocol`

For each agent turn, hooks execute in this sequence:

  1. before_agent (all middleware)
     ├─ PatchToolCallsMiddleware: Fix dangling tool calls
     └─ SummarizationMiddleware: Summarize if needed

  2. wrap_model_call (all middleware)  
     ├─ FilesystemMiddleware: Inject filesystem instructions
     ├─ SubAgentMiddleware: Inject subagent instructions
     └─ AnthropicPromptCachingMiddleware: Add cache headers

  3. [Model generates response with tool calls]

  4. wrap_tool_call (all middleware, for each tool call)
     ├─ FilesystemMiddleware: Evict large results to files
     └─ HumanInTheLoopMiddleware: Pause for approval if configured

## Writing Custom Instructions

When building a deepagent, you provide a `system_prompt` parameter to `create_deep_agent()`. This custom prompt is **appended to** default instructions that are automatically injected by middleware. Understanding this layering is crucial for writing effective custom instructions.

### Default Instructions (Injected by Middleware)

The middleware automatically adds instructions about the standard tools. Your custom instructions should **complement, not duplicate** these defaults:

#### From TodoListMiddleware
- Explains when to use `write_todos` and `read_todos`
- Guidance on marking tasks completed
- Best practices for todo list management
- When NOT to use todos (simple tasks)

#### From FilesystemMiddleware
- Lists all filesystem tools (`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `execute`*)
- Explains that file paths must start with `/`
- Describes each tool's purpose and parameters
- Notes about context offloading for large tool results

#### From SubAgentMiddleware
- Explains the `task()` tool for delegating to sub-agents
- When to use sub-agents vs when NOT to use them
- Guidance on parallel execution
- Subagent lifecycle (spawn → run → return → reconcile)

### Writing Effective Custom Instructions

**Do:**
- ✅ Define domain-specific workflows (e.g., research methodology, data analysis steps)
- ✅ Provide concrete examples for your use case
- ✅ Add specialized guidance (e.g., "batch similar research tasks into a single TODO")
- ✅ Define stopping criteria and resource limits
- ✅ Explain how tools work together in your workflow

**Don't:**
- ❌ Re-explain what standard tools do (already covered by middleware)
- ❌ Duplicate middleware instructions about tool usage
- ❌ Contradict default instructions (work with them, not against them)

### Example: Research Agent Custom Instructions

The deep research quickstart demonstrates this pattern well:

```python
RESEARCH_WORKFLOW_INSTRUCTIONS = """# Research Workflow
Follow this workflow for all research requests:
1. **Save the request**: Use write_file() to save question to `/research_request.md`
2. **Plan**: Create todo list with write_todos
3. **Research**: Delegate tasks to sub-agents using task()
4. **Synthesize**: Read saved request to confirm you've addressed all aspects
5. **Respond**: Provide comprehensive answer

## Research Planning Guidelines
- Batch similar research tasks into a single TODO
- For simple queries, use 1 sub-agent
- For comparisons, delegate to multiple parallel sub-agents
"""
```

This custom instruction:
- ✅ Defines a specific workflow for the research domain
- ✅ Shows how standard tools (`write_file`, `write_todos`, `task`) work together
- ✅ Provides concrete scaling rules
- ❌ Doesn't re-explain what each tool does (middleware handles that)

### Viewing the Combined Prompt

To see the full prompt (default + custom instructions), you can inspect the middleware system prompt after agent creation, or use the helper function from the examples:

```python
from utils import show_prompt

show_prompt(INSTRUCTIONS)  # Shows your custom instructions
# The middleware adds its instructions automatically at runtime
```

## Quickstarts

For each quickstart, we can amend the core deepagent harness with any custom tools, middleware, and / or instructions.

| Quickstart Name | Location | Description |
|----------------|----------|-------------|
| Deep Research | `examples/deep_research/` | A research agent that conducts multi-step web research using Tavily for URL discovery, fetches full webpage content, and coordinates work through parallel sub-agents and strategic reflection |

### Deep Research  

#### Instructions

The deep research agent uses streamlined custom instructions defined in `deep_research/research_agent/prompts.py` that complement (rather than duplicate) the default middleware instructions:

| Instruction Set | Purpose |
|----------------|---------|
| `RESEARCH_WORKFLOW_INSTRUCTIONS` | Defines the 5-step research workflow: save request → plan with TODOs → delegate to sub-agents → synthesize → respond. Includes research-specific planning guidelines like batching similar tasks and scaling rules for different query types. |
| `SUBAGENT_DELEGATION_INSTRUCTIONS` | Provides concrete delegation strategies with examples: simple queries use 1 sub-agent, comparisons use 1 per element, multi-faceted research uses 1 per aspect. Sets limits on parallel execution (max 3 concurrent) and iteration rounds (max 3). |
| `RESEARCHER_INSTRUCTIONS` | Guides individual research sub-agents to conduct focused web searches. Includes hard limits (2-3 searches for simple queries, max 5 for complex), emphasizes using `think_tool` after each search for strategic reflection, and defines stopping criteria. |

#### Tools

The deep research agent adds the following custom tools beyond the core deepagent tools:

| Tool Name | Description |
|-----------|-------------|
| `tavily_search` | Web search tool that uses Tavily purely as a URL discovery engine. Performs searches using Tavily API to find relevant URLs, fetches full webpage content via HTTP with proper User-Agent headers (avoiding 403 errors), converts HTML to markdown, and returns the complete content without summarization to preserve all information for the agent's analysis. |
| `think_tool` | Strategic reflection mechanism that helps the agent pause and assess progress between searches, analyze findings, identify gaps, and plan next steps. |
