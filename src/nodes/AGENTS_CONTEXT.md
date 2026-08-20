# src/nodes/ AGENTS_CONTEXT.md

## Purpose
LangGraph nodes for the SQL copilot agent workflow - database selection, intent classification, authorization, the SQL generation agent loop, validation, execution, and result analysis.

## Key Files

- **database_checker.py**: Verify if the database correspond to the field of the user's question.
- **intent_classifier.py**: Classifies user intent (query vs modification)
- **role_authorizer.py**: Checks if user has authorization for modification operations
- **sql_agent.py**: The SQL generation agent loop — four nodes
  (`SQLAgentLLMNode`, `SQLAgentToolsNode`, `SQLAgentClarifyNode`,
  `SQLAgentFinalizeNode`), wired into `graph.py` in place of `sql_generator.py`
  and `sql_fallback_regenerator.py` (both retired from the graph).
- **sql_generator.py**: Single-shot NL→SQL node. No longer wired into the
  graph (replaced by `sql_agent.py`); kept in the repo but unused by
  `build_sql_agent_graph`.
- **sql_validator.py**: Validates generated SQL for safety before execution
- **sql_modification_validator.py**: Manages interruption/confirmation workflow for modifications
- **sql_executor.py**: Executes validated SQL against SQLite database; on
  failure, feeds the error back into the agent `messages` transcript and
  increments `retry_count` so `sql_agent_llm` can repair it (D6) — there is no
  dedicated regeneration node anymore.
- **result_analyst.py**: Transforms query results into user-facing analysis

## Important Exports

- DatabaseCheckerNode - Check that the question correspond to the selected database
- IntentClassifierNode - Classifies user intent
- RoleAuthorizerNode - Authorizes modification operations
- SQLAgentLLMNode - Seeds/continues the agent transcript and calls the tool-calling model
- SQLAgentToolsNode - Dispatches inspect_schema/run_readonly_probe tool calls; rejects
  mixed-batch ask_user calls (D3) and enforces the probe budget
- SQLAgentClarifyNode - Interrupts for user input when ask_user is the sole tool call
- SQLAgentFinalizeNode - Extracts the final SQL + rationale, or resets state for a repair pass
- SQLAgentBudgetExhaustedNode - Asks the (unbound) model to explain what it
  learned when the iteration cap is hit while tools are still pending
- SQLValidatorNode - Validates SQL safety
- SQLModificationValidatorNode - Manages modification confirmation
- SQLExecutorNode - Executes SQL queries; formats a repair-turn HumanMessage on failure
- ResultAnalystNode - Analyzes query results
- SQLGeneratorNode - Unused by the graph; kept for potential reuse

## Node Execution Flow

1. **database_checker** → Verify that if the question correspond to the content of the database
2. **intent_classifier** → Classifies request as query or modification
3. **role_authorizer** → Checks user permissions (modifications only) or aborts
4. **sql_agent_llm** ⇄ **sql_agent_tools** → Agent loop: the model calls
   `inspect_schema`/`run_readonly_probe` as many times as it needs, looping
   back to sql_agent_llm after each batch, until it stops requesting tools
5. **sql_agent_clarify** → **Interruption point**: reached only when the model's
   sole tool call is `ask_user`; pauses and waits for the user's answers, then
   loops back to sql_agent_llm
6. **sql_agent_finalize** → Extracts the final SQL statement (+ optional
   rationale) from the transcript; on a repair pass, resets validation/execution
   state the same way the retired fallback regenerator used to
7. **sql_agent_budget_exhausted** → Reached only if the model still wants to
   call tools once the iteration cap is hit (a clean finalize is never
   blocked); asks the unbound model to explain its progress, then ends
8. **sql_validator** → Validates SQL for query (SELECT only, no forbidden keywords)
9. **modification_validator** → **Interruption point**: pauses execution and waits for user confirmation before modifications
10. **sql_executor** → Executes validated SQL with result limit; on failure,
    routes back to sql_agent_llm (bounded by `retry_count` / `max_retries`)
11. **result_analyst** → Produces user-friendly analysis

### Interruption Workflow

Two nodes use LangGraph's interruption feature, sharing the same resume/session plumbing:

- **sql_agent_clarify**: pauses when the agent needs a business detail it
  cannot derive from the schema or a probe. User answers are appended to the
  transcript as a `ToolMessage` and the loop continues.
- **modification_validator**: pauses after a modification statement is
  finalized, before it executes.
  - **User action**: Review the modification query and confirm/cancel
  - **If confirmed**: Execution resumes to sql_executor
  - **If cancelled**: Graph aborts with an error

This ensures users can review potentially destructive operations before they execute.

## Key Dependencies

- state.py - SQLAgentState for node communication (messages transcript + agent budget/status keys)
- tools/database.py - SQLiteDatabase, QueryResult, and full-schema formatting
- tools/sql_safety.py - SQLSafetyValidator
- tools/agent_tools.py - inspect_schema / ask_user / run_readonly_probe, built by build_agent_tools
- utils/llm.py - LLM response extraction utilities
- utils/nodes.py - Prompt loading/rendering, LLM and ToolCallingChatModel protocols
- prompts/ - Jinja2 template files

## When to Read Files

- **database_checker.py**: When modifying database verification logic
- **intent_classifier.py**: When changing intent classification prompt or behavior
- **role_authorizer.py**: When updating authorization checks
- **sql_agent.py**: When changing the agent loop's nodes, budgets, or final-SQL extraction
- **sql_validator.py**: When updating SQL safety validation rules
- **sql_modification_validator.py**: When modifying modification confirmation workflow
- **sql_executor.py**: When adjusting query execution parameters or the D6 repair hand-off
- **result_analyst.py**: When changing result analysis or LLM usage

## Related Contexts

- **src/AGENTS_CONTEXT.md** - Root graph and state definitions
- **src/tools/AGENTS_CONTEXT.md** - Database, SQL safety, and agent tools
- **src/prompts/AGENTS_CONTEXT.md** - Jinja2 template files
