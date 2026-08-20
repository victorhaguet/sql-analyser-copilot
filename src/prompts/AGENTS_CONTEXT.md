# src/prompts/ AGENTS_CONTEXT.md

## Purpose
Jinja2 prompt templates for SQL copilot nodes.

## Key Files

- **database_checker.j2**: Template for the database-checker node — verifies
  the question matches the selected database's schema
- **intent_classifier.j2**: Template for intent classification node
- **result_analyst_query.j2**: Template for analyzing SELECT results
- **result_analyst_modification.j2**: Template for summarizing a completed
  modification (no `result_payload` — modifications have no rows to show)
- **sql_agent.j2**: System prompt seeding the SQL generation agent loop
  (`nodes/sql_agent.py`) — states the three tools, the probe-before-finalize
  rule, the `ask_user`-alone rule (D3), and the INSERT/UPDATE/DELETE
  pre-conditions. Also covers D6 repair turns (the retired
  `sql_fallback_regenerator.j2`'s rules were folded in here). Unlike every
  other node, this file has **no hardcoded fallback string** —
  `SQLAgentLLMNode` raises if it is missing or empty rather than silently
  degrading to a weaker built-in prompt.
- **sql_generator.j2**: Template for the retired single-shot `sql_generator.py`
  node. Not rendered by `build_sql_agent_graph`; kept only because
  `sql_generator.py` itself is kept unused in the repo.

Notes: `role_authorizer.py` and `sql_validator.py` are pure code (no LLM call),
so they have no `.j2` template here.

## Key Features

- Jinja2 templating with context variables
- Fallback prompts defined in node code (except `sql_agent.j2`, which has none)
- Agent-loop system prompt covering tool use, clarification, and repair rules
- Modification confirmation workflows

## Context Variables

- **database_checker**: `question`, `schema_overview`
- **intent_classifier**: `question`
- **result_analyst_query**: `question`, `sql`, `result_payload`
- **result_analyst_modification**: `question`, `sql`
- **sql_agent**: `intent`, `schema_overview`, `schema_truncated` (the question
  itself is a separate seeded `HumanMessage`, not a template variable)
- **sql_generator** (unused): `question`, `schema_overview`

## When to Read Files

- **database_checker.j2**: When modifying the database-match prompt
- **intent_classifier.j2**: When updating intent classification prompt
- **result_analyst_query.j2**: When changing how SELECT results are explained
- **result_analyst_modification.j2**: When changing how completed
  modifications are summarized
- **sql_agent.j2**: When changing the agent loop's tool-use, clarification, or
  repair rules — this is the prompt that actually drives SQL generation today

## Related Contexts

- **src/nodes/AGENTS_CONTEXT.md** - Nodes load and render these templates
