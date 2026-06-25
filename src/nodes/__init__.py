"""Node utilities."""

from state import SQLAgentState
from tools.database import RegisteredDatabase, SQLiteDatabase


def get_database(state: SQLAgentState, default: SQLiteDatabase, database_catalog: list[RegisteredDatabase] | None = None) -> SQLiteDatabase:
    """Get database from state or return default.

    Args:
        state (SQLAgentState): State of the graph
        default (SQLiteDatabase): Default database
        database_catalog (list[RegisteredDatabase] | None): Database catalog for lookup

    Returns:
        SQLiteDatabase: Current database in the state
    """    
    selected_database_name = state.get("selected_database")
    if selected_database_name and database_catalog:
        for entry in database_catalog:
            if entry.name == selected_database_name:
                return entry.database
    return default
