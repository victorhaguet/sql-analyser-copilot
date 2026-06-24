# src/nodes/ AGENTS_CONTEXT.md

## Purpose
LangGraph nodes for the SQL copilot agent workflow - database selection, SQL generation, validation, execution, and result analysis.

## Key Files

- **database_selector.py**: Selects most relevant database for user question
- **sql_generator.py**: Generates SQL from natural language question
- **sql_validator.py**: Validates generated SQL for safety before execution
- **sql_executor.py**: Executes validated SQL against SQLite database
- **result_analyst.py**: Transforms query results into user-facing analysis

## Important Exports

- DatabaseSelectorNode - Routes question to appropriate database
- SQLGeneratorNode - Converts natural language to SQL
- SQLValidatorNode - Validates SQL safety
- SQLExecutorNode - Executes SQL queries
- ResultAnalystNode - Analyzes query results

## Node Execution Flow

1. **database_selector** → Routes question to database (or aborts)
2. **sql_generator** → Generates SQL from question + schema
3. **sql_validator** → Validates SQL (SELECT only, no forbidden keywords)
4. **sql_executor** → Executes validated SQL with result limit
5. **result_analyst** → Produces user-friendly analysis

## Key Dependencies

- state.py - SQLAgentState for node communication
- tools/database.py - SQLiteDatabase and QueryResult
- tools/sql_safety.py - SQLSafetyValidator
- utils/llm.py - LLM response extraction utilities
- utils/nodes.py - Prompt loading and rendering
- prompts/ - Jinja2 template files

## When to Read Files

- **database_selector.py**: When modifying database routing logic
- **sql_generator.py**: When changing SQL generation prompt or behavior
- **sql_validator.py**: When updating SQL safety validation rules
- **sql_executor.py**: When adjusting query execution parameters
- **result_analyst.py**: When changing result analysis or LLM usage

## Related Contexts

- **src/AGENTS_CONTEXT.md** - Root graph and state definitions
- **src/tools/AGENTS_CONTEXT.md** - Database and SQL safety tools
- **src/prompts/AGENTS_CONTEXT.md** - Jinja2 template files
