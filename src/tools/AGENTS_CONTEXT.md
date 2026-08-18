# src/tools/ AGENTS_CONTEXT.md

## Purpose
Core utility modules for database operations and SQL safety validation.

## Key Files

- **database.py**: SQLiteDatabase class, query execution, schema introspection
- **sql_safety.py**: SQL safety validator - prevents dangerous queries
- **exceptions.py**: Custom exception classes for SQL safety errors
- **_helpers.py**: Internal helpers for SQL masking and parameter extraction
- **agent_tools.py**: `@tool` callables for the SQL generation agent loop
  (`inspect_schema`, `ask_user`, `run_readonly_probe`), built by
  `build_agent_tools(database, validator, limits)`

## Important Exports

- SQLiteDatabase - Read-only SQLite database wrapper
- SQLSafetyValidator - Validates SQL queries for safety
- QueryResult - Dataclass for query results
- RegisteredDatabase - Database entry with routing metadata
- DatabaseError, SQLSafetyError - Custom exceptions
- build_agent_tools, AgentToolLimits - Factory + tunables for the agent's tools

## Key Features

- Read-only SQLite access with parameterization
- Schema introspection (list tables, get schema, preview data)
- Relational introspection: `get_foreign_keys`, `get_indexes`,
  `list_referencing_tables` (incoming FKs), `format_table_detail` (columns with
  NOT NULL/DEFAULT/PK, FKs in both directions, unique indexes, optional sample
  rows), `format_full_schema` (every table via `format_table_detail`, with a
  `max_chars` guard that falls back to `format_database_schema`)
- SQL safety validation (SELECT only, no forbidden keywords)
- Result limiting and truncation detection
- Multiple database support with routing catalog
- Agent tools: schema inspection (with error text for unknown tables, never
  raises), interrupt-based user clarification (`ask_user`, must be called alone
  per D3), and a read-only probe of arbitrary SELECTs. The probe is read-only
  by construction — a `mode=ro` SQLite URI connection plus the safety validator
  are two independent guards, so a write fails even if the validator is bypassed.

## When to Read Files

- **database.py**: When adding database features or modifying query execution
- **sql_safety.py**: When updating SQL safety validation rules
- **exceptions.py**: When extending error handling
- **_helpers.py**: When modifying SQL masking logic or parameter extraction
- **agent_tools.py**: When changing the agent's tool set or its row/timeout limits

## Related Contexts

- **src/nodes/AGENTS_CONTEXT.md** - Nodes use database and safety tools
- **src/AGENTS_CONTEXT.md** - Core execution uses these tools
