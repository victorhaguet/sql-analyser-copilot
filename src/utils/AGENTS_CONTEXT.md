# src/utils/ AGENTS_CONTEXT.md

## Purpose
Shared utilities used across the SQL copilot application.

## Key Files

- **llm.py**: LLM response processing (extract text, strip code fences)
- **nodes.py**: Prompt loading and Jinja2 template rendering
- **paths.py**: Path utilities (DEFAULT_DB_PATH configuration)

## Important Exports

- extract_text_from_response() - Extract text from LLM responses
- strip_code_fences() - Remove Markdown code fences and language prefixes
- load_prompt() - Load Jinja2 prompt templates with fallback
- render_prompt() - Render Jinja2 templates with context
- DEFAULT_DB_PATH - Default SQLite database path
- LLM - Protocol for the invoke(prompt: str) -> Any nodes (generator, analyst, intent)
- ToolCallingChatModel - Protocol for bind_tools-capable chat models, used by the
  SQL generation agent loop (`ChatOpenAI` satisfies it structurally)

## When to Read Files

- **llm.py**: When modifying LLM response parsing
- **nodes.py**: When changing prompt loading or template rendering
- **paths.py**: When updating default paths or auto-discovery logic

## Related Contexts

- **src/nodes/AGENTS_CONTEXT.md** - Nodes use these utilities for prompts
- **src/AGENTS_CONTEXT.md** - Core execution depends on utilities
