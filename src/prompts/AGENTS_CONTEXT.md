# src/prompts/ AGENTS_CONTEXT.md

## Purpose
Jinja2 prompt templates for SQL copilot nodes.

## Key Files

- **database_selector.j2**: Template for database selection node
- **sql_generator.j2**: Template for SQL generation node
- **result_analyst.j2**: Template for result analysis node

## Key Features

- Jinja2 templating with context variables
- Fallback prompts defined in node code
- Natural language to SQL conversion templates

## Context Variables

- **database_selector**: `question`, `catalog_payload`
- **sql_generator**: `question`, `schema_overview`
- **result_analyst**: `question`, `sql`, `result_payload`

## When to Read Files

- **database_selector.j2**: When modifying database selection prompt
- **sql_generator.j2**: When updating SQL generation prompt
- **result_analyst.j2**: When changing result analysis prompt

## Related Contexts

- **src/nodes/AGENTS_CONTEXT.md** - Nodes load and render these templates
