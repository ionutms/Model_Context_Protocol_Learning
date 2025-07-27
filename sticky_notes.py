import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Sticky Notes")


NOTES_FILE = os.path.join(os.path.dirname(__file__), "notes.txt")


def ensure_file_exists():
    """Ensure the notes file exists."""
    if not os.path.exists(NOTES_FILE):
        with open(NOTES_FILE, "w") as file:
            file.write("")


# The dockstring is important for the tool to be recognized by the MCP server.
@mcp.tool()
def add_note(note: str) -> str:
    """Add a new note to the note file.

    Args:
        note (str): The note to add.

    Returns:
        str: Confirmation message.
    """
    ensure_file_exists()
    with open(NOTES_FILE, "a") as file:
        file.write(note + "\n")
    return f"Note added: {note}"


# The dockstring is important for the tool to be recognized by the MCP server.
@mcp.tool()
def list_notes() -> str:
    """List all notes in the note file.

    Returns:
        str: All notes as a single string.
    """
    ensure_file_exists()
    with open(NOTES_FILE, "r") as file:
        notes = file.readlines()
    return "".join(notes) if notes else "No notes found."


# The dockstring is important for the tool to be recognized by the MCP server.
@mcp.resource("notes://latest")
def get_note() -> str:
    """Get the latest note from the note file.

    Returns:
        str: The latest note or a message if no notes are available.
    """
    ensure_file_exists()
    with open(NOTES_FILE, "r") as file:
        lines = file.readlines()
    return lines[-1].strip() if lines else "No notes available."


# The dockstring is important for the tool to be recognized by the MCP server.
@mcp.prompt()
def prompt_note() -> str:
    """Generate a prompt to summarize the latest notes.

    Returns:
        str: A prompt summarizing the latest notes.
    """
    ensure_file_exists()
    with open(NOTES_FILE, "r") as file:
        notes = file.readlines()
    if not notes:
        return "No notes available. Please add a note first."
    return f"Summarize the latest notes: {notes}"
