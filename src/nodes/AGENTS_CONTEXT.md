# src/nodes/ AGENTS_CONTEXT.md

## Purpose
LangGraph nodes for the SQL copilot agent workflow - database selection, intent classification, authorization, SQL generation, validation, execution, and result analysis.

## Key Files

- **database_selector.py**: Selects most relevant database for user question
- **intent_classifier.py**: Classifies user intent (query vs modification)
- **role_authorizer.py**: Checks if user has authorization for modification operations
- **sql_generator.py**: Generates SQL from natural language question
- **sql_validator.py**: Validates generated SQL for safety before execution
- **sql_modification_validator.py**: Manages interruption/confirmation workflow for modifications
- **sql_executor.py**: Executes validated SQL against SQLite database
- **result_analyst.py**: Transforms query results into user-facing analysis

## Important Exports

- DatabaseSelectorNode - Routes question to appropriate database
- IntentClassifierNode - Classifies user intent
- RoleAuthorizerNode - Authorizes modification operations
- SQLGeneratorNode - Converts natural language to SQL
- SQLValidatorNode - Validates SQL safety
- SQLModificationValidatorNode - Manages modification confirmation
- SQLExecutorNode - Executes SQL queries
- ResultAnalystNode - Analyzes query results

## Node Execution Flow

1. **database_selector** → Routes question to database (or aborts on failure)
2. **intent_classifier** → Classifies request as query or modification
3. **role_authorizer** → Checks user permissions (modifications only) or aborts
4. **sql_generator** → Generates SQL from question + schema
5. **sql_validator** → Validates SQL for query (SELECT only, no forbidden keywords)
6. **modification_validator** → **Interruption point**: pauses execution and waits for user confirmation before modifications
7. **sql_executor** → Executes validated SQL with result limit
8. **result_analyst** → Produces user-friendly analysis

### Interruption Workflow

The **modification_validator** node uses LangGraph's interruption feature to pause execution:

- **For modifications**: The graph pauses after SQL generation and waits for user confirmation
- **User action**: Review the modification query and confirm/cancel
- **If confirmed**: Execution resumes to sql_executor
- **If cancelled**: Graph aborts with an error

This ensures users can review potentially destructive operations before they execute.

## Key Dependencies

- state.py - SQLAgentState for node communication
- tools/database.py - SQLiteDatabase and QueryResult
- tools/sql_safety.py - SQLSafetyValidator
- utils/llm.py - LLM response extraction utilities
- utils/nodes.py - Prompt loading and rendering
- prompts/ - Jinja2 template files

## When to Read Files

- **database_selector.py**: When modifying database routing logic
- **intent_classifier.py**: When changing intent classification prompt or behavior
- **role_authorizer.py**: When updating authorization checks
- **sql_generator.py**: When changing SQL generation prompt or behavior
- **sql_validator.py**: When updating SQL safety validation rules
- **sql_modification_validator.py**: When modifying modification confirmation workflow
- **sql_executor.py**: When adjusting query execution parameters
- **result_analyst.py**: When changing result analysis or LLM usage

## Related Contexts

- **src/AGENTS_CONTEXT.md** - Root graph and state definitions
- **src/tools/AGENTS_CONTEXT.md** - Database and SQL safety tools
- **src/prompts/AGENTS_CONTEXT.md** - Jinja2 template files
