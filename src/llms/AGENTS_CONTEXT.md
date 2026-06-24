# src/llms/ AGENTS_CONTEXT.md

## Purpose
LLM provider integrations - currently OpenAI-compatible API wrapper.

## Key Files

- **openai_compatible.py**: OpenAICompatibleResponsesModel - adapter for OpenAI-compatible APIs

## Important Exports

- OpenAICompatibleResponsesModel - LLM adapter implementing the LLM protocol

## Key Features

- OpenAI-compatible API support (base_url, api_key configuration)
- Chat completion interface
- Model name configuration via environment variables

## When to Read Files

- **openai_compatible.py**: When modifying LLM adapter or adding new providers

## Related Contexts

- **src/nodes/AGENTS_CONTEXT.md** - Nodes use LLM for SQL generation and analysis
- **src/AGENTS_CONTEXT.md** - LLM configuration in config.py
