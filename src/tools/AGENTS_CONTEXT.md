# src/tools/ AGENTS_CONTEXT.md

## Purpose
Core utility modules for database operations and SQL safety validation.

## Key Files

- **database.py**: SQLiteDatabase class, query execution, schema introspection
- **sql_safety.py**: SQL safety validator - prevents dangerous queries
- **exceptions.py**: Custom exception classes for SQL safety errors
- **_helpers.py**: Internal helpers for SQL masking and parameter extraction

## Important Exports

- SQLiteDatabase - Read-only SQLite database wrapper
- SQLSafetyValidator - Validates SQL queries for safety
- QueryResult - Dataclass for query results
- RegisteredDatabase - Database entry with routing metadata
- DatabaseError, SQLSafetyError - Custom exceptions

## Key Features

- Read-only SQLite access with parameterization
- Schema introspection (list tables, get schema, preview data)
- SQL safety validation (SELECT only, no forbidden keywords)
- Result limiting and truncation detection
- Multiple database support with routing catalog

## When to Read Files

- **database.py**: When adding database features or modifying query execution
- **sql_safety.py**: When updating SQL safety validation rules
- **exceptions.py**: When extending error handling
- **_helpers.py**: When modifying SQL masking logic or parameter extraction

## Related Contexts

- **src/nodes/AGENTS_CONTEXT.md** - Nodes use database and safety tools
- **src/AGENTS_CONTEXT.md** - Core execution uses these tools
