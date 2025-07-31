import ast
import glob
import os
import subprocess
import tokenize
from io import StringIO

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Python File Reader")

INPUT_DIR = os.path.join(os.path.dirname(__file__), "to_analize")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "analize_done")


def ensure_directories_exist():
    """Ensure both input and output directories exist."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def remove_docstrings_and_comments(code: str) -> str:
    """Remove all docstrings and comments from Python code."""
    tree = ast.parse(code)
    lines = code.split("\n")
    docstring_ranges = []
    comment_lines = set()

    # Find module docstring
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        start_line = tree.body[0].lineno - 1
        end_line = (
            tree.body[0].end_lineno - 1
            if tree.body[0].end_lineno
            else start_line
        )
        docstring_ranges.append({
            "start": start_line,
            "end": end_line,
            "type": "module",
        })

    # Find function/class docstrings
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
        ):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                docstring_node = node.body[0]
                start_line = docstring_node.lineno - 1
                end_line = (
                    docstring_node.end_lineno - 1
                    if docstring_node.end_lineno
                    else start_line
                )
                docstring_ranges.append({
                    "start": start_line,
                    "end": end_line,
                    "type": "function_class",
                    "is_only_body": len(node.body) == 1,
                })

    # Find comments
    string_io = StringIO(code)
    tokens = list(tokenize.generate_tokens(string_io.readline))
    for token in tokens:
        if token.type == tokenize.COMMENT:
            comment_lines.add(token.start[0] - 1)

    # Combine and sort ranges
    all_ranges = docstring_ranges + [
        {"start": line, "end": line, "type": "comment"}
        for line in comment_lines
    ]
    all_ranges.sort(key=lambda x: x["start"], reverse=True)

    # Remove docstrings and comments
    for range_info in all_ranges:
        start_line = range_info["start"]
        end_line = range_info["end"]

        if start_line < len(lines):
            indent_level = len(lines[start_line]) - len(
                lines[start_line].lstrip()
            )
        else:
            indent_level = 0

        del lines[start_line : end_line + 1]

        # Remove empty lines after module docstring
        if range_info.get("type") == "module":
            while start_line < len(lines) and not lines[start_line].strip():
                del lines[start_line]

        # Add 'pass' if function/class body is empty after removing docstring
        if range_info.get("is_only_body", False):
            lines.insert(start_line, " " * indent_level + "pass")

    return "\n".join(lines)


@mcp.tool()
def process_python_file(filename: str) -> str:
    """
    Process a Python file by removing comments and docstrings.

    This function reads a Python file from the input directory, removes all
    comments and docstrings from the code, and saves the cleaned version to
    the output directory.
    The original file structure and functionality are preserved while making
    the code more compact for analysis purposes.

    Args:
        filename (str): The name of the Python file to process.

    Returns:
        str:
            A success message indicating the file was processed and saved,
            or an error message if the operation failed.
    """
    ensure_directories_exist()

    if not filename.endswith(".py"):
        return (
            f"Error: '{filename}' is not a Python file. "
            "Only .py files are supported."
        )

    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(input_path):
        return f"File '{filename}' not found in the to_analize directory."

    with open(input_path, "r", encoding="utf-8") as file:
        original_code = file.read()

    processed_code = remove_docstrings_and_comments(original_code)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(processed_code)

    return (
        f"Successfully processed '{filename}' "
        f"and saved to {OUTPUT_DIR} directory."
    )


@mcp.tool()
def process_all_python_files() -> str:
    """Process all Python files in the input directory.

    This function discovers all Python files in the input directory and
    processes each one by removing comments and docstrings.
    It provides a batch processing capability for multiple files at once,
    with individual status reporting for each file processed.

    Returns:
        str:
            A formatted report showing the number of files processed and the
            individual results for each file. If no Python files are found,
            returns an appropriate message.
    """
    ensure_directories_exist()

    python_files = glob.glob(os.path.join(INPUT_DIR, "*.py"))
    if not python_files:
        return "No Python files found to process."

    results = []
    for file_path in python_files:
        filename = os.path.basename(file_path)
        result = process_python_file(filename)
        results.append(f"- {result}")

    return f"Processed {len(python_files)} files:\n" + "\n".join(results)


@mcp.tool()
def list_python_files() -> str:
    """List all Python files available in the input directory.

    This function scans the input directory and returns a formatted list of
    all Python files that are available for processing.
    This is useful for checking what files can be analyzed before running
    processing operations.

    Returns:
        str:
            A formatted list of Python file names from in the input directory.
            If no Python files are found, returns a message indicating the
            directory is empty.
    """
    ensure_directories_exist()
    python_files = glob.glob(os.path.join(INPUT_DIR, "*.py"))
    if not python_files:
        return "No Python files found in the directory."

    file_names = [os.path.basename(file) for file in python_files]
    return "Python files found:\n" + "\n".join(
        f"- {name}" for name in file_names
    )


@mcp.tool()
def read_processed_file(filename: str) -> str:
    """Read and display the contents of a processed Python file.

    This function reads a previously processed Python file from the output
    directory and returns its contents. The processed file will have all
    comments and docstrings removed, showing only the core code structure.
    This is useful for reviewing the results of the processing operation.

    Args:
        filename (str):
            The name of the processed Python file to read.

    Returns:
        str:
            The complete contents of the processed file with a header
            indicating the filename, or an error message if the file
            cannot be read.
    """
    ensure_directories_exist()

    if not filename.endswith(".py"):
        return (
            f"Error: '{filename}' is not a Python file. "
            "Only .py files are supported."
        )

    output_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(output_path):
        return (
            f"Processed file '{filename}' not found "
            f"in the {OUTPUT_DIR} directory."
        )

    with open(output_path, "r", encoding="utf-8") as file:
        content = file.read()

    return f"Contents of processed file '{filename}':\n\n{content}"


@mcp.tool()
def list_processed_files() -> str:
    """List all processed Python files in the output directory.

    This function scans the output directory and returns a formatted list of
    all processed Python files that are available. These are files that have
    been successfully processed to remove comments and docstrings.
    This helps track which files have been analyzed and are ready for review.

    Returns:
        str:
            A formatted list of processed Python file names found in the
            output directory. If no processed files are found, returns a
            message indicating the directory is empty.
    """
    ensure_directories_exist()
    python_files = glob.glob(os.path.join(OUTPUT_DIR, "*.py"))
    if not python_files:
        return (
            "No processed Python files found in the analize_done directory."
        )

    file_names = [os.path.basename(file) for file in python_files]
    return "Processed Python files found:\n" + "\n".join(
        f"- {name}" for name in file_names
    )


@mcp.tool()
def save_processed_file(filename: str, content: str) -> str:
    """Save content to a Python file in the output directory.

    This function allows you to save modified or new content to a Python file
    in the output directory. This is useful for saving edited versions of
    processed files or creating new files based on analysis results.
    The function will validate that the filename has a .py extension and
    create the output directory if it doesn't exist.

    Args:
        filename (str): The name of the Python file to save.
        content (str): The Python code content to save to the file.

    Returns:
        str:
            A success message indicating the file was saved successfully,
            or an error message if the operation failed.
    """
    ensure_directories_exist()

    if not filename.endswith(".py"):
        return (
            f"Error: '{filename}' is not a Python file. "
            "Only .py files are supported."
        )

    output_path = os.path.join(OUTPUT_DIR, filename)

    try:
        # Validate that the content is valid Python syntax
        ast.parse(content)

        with open(output_path, "w", encoding="utf-8") as file:
            file.write(content)

        return (
            f"Successfully saved '{filename}' to the {OUTPUT_DIR} directory."
        )

    except SyntaxError as e:
        return (
            f"Error: Invalid Python syntax in content for '{filename}'. "
            f"Syntax error: {str(e)}"
        )
    except Exception as e:
        return f"Error saving '{filename}': {str(e)}"


@mcp.tool()
def run_ruff_check(filename: str) -> str:
    """Run ruff check on a single processed Python file.

    Args:
        filename (str): The name of the Python file to check with ruff.

    Returns:
        str: Ruff check results or error message.
    """
    ensure_directories_exist()

    if not filename.endswith(".py"):
        return f"Error: '{filename}' is not a Python file."

    output_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(output_path):
        return f"Error: File '{filename}' not found in output directory."

    try:
        result = subprocess.run(
            ["ruff", "check", output_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = result.stdout + result.stderr

        if result.returncode == 0 and not output.strip():
            return f"✅ {filename}: No issues found"
        elif output.strip():
            return f"⚠️ {filename} issues:\n{output.strip()}"
        else:
            return f"❌ {filename}: Check failed (code {result.returncode})"

    except subprocess.TimeoutExpired:
        return f"Error: Ruff check timed out for '{filename}'"
    except FileNotFoundError:
        return "Error: ruff not found. Install with: pip install ruff"
    except Exception as e:
        return f"Error running ruff on '{filename}': {str(e)}"
