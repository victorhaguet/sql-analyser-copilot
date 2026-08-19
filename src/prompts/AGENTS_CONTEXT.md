# src/prompts/ AGENTS_CONTEXT.md

## Purpose
Jinja2 prompt templates for SQL copilot nodes.

## Key Files

- **database_selector.j2**: Template for database selection node
- **intent_classifier.j2**: Template for intent classification node
- **role_authorizer.j2**: Template for role authorization node
- **sql_generator.j2**: Template for SQL generation node
- **sql_validator.j2**: Template for SQL safety validation node
- **result_analyst.j2**: Template for result analysis node
- **sql_agent.j2**: System prompt seeding the SQL generation agent loop
  (`nodes/sql_agent.py`) — states the three tools, the probe-before-finalize
  rule (D6), and the INSERT/UPDATE/DELETE pre-conditions. Also covers D6 repair
  turns (the retired `sql_fallback_regenerator.j2`'s rules were folded in here).
  Unlike every other node, this file has **no hardcoded fallback string** —
  `SQLAgentLLMNode` raises if it is missing or empty rather than silently
  degrading to a weaker built-in prompt.

## Key Features

- Jinja2 templating with context variables
- Fallback prompts defined in node code
- Natural language to SQL conversion templates
- Modification confirmation workflows

## Context Variables

- **database_selector**: `question`, `catalog_payload`
- **intent_classifier**: `question`
- **role_authorizer**: `question`, `user_role`
- **sql_generator**: `question`, `schema_overview`
- **sql_validator**: `question`, `sql`, `schema_overview`
- **result_analyst**: `question`, `sql`, `result_payload`
- **sql_agent**: `intent`, `schema_overview`, `schema_truncated` (the question
  itself is a separate seeded `HumanMessage`, not a template variable)

## When to Read Files

- **database_selector.j2**: When modifying database selection prompt
- **intent_classifier.j2**: When updating intent classification prompt
- **role_authorizer.j2**: When changing authorization prompt
- **sql_generator.j2**: When updating SQL generation prompt
- **sql_validator.j2**: When modifying SQL validation prompt
- **result_analyst.j2**: When changing result analysis prompt

## Related Contexts

- **src/nodes/AGENTS_CONTEXT.md** - Nodes load and render these templates
