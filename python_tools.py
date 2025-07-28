import ast
import glob
import json
import os
import tokenize
from io import StringIO
from typing import List

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


def extract_function_signature(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> str:
    """Extract a clean function signature with type hints."""
    args = []

    # Handle regular arguments
    for arg in node.args.args:
        arg_str = arg.arg
        if arg.annotation:
            arg_str += f": {ast.unparse(arg.annotation)}"
        args.append(arg_str)

    # Handle *args
    if node.args.vararg:
        vararg_str = f"*{node.args.vararg.arg}"
        if node.args.vararg.annotation:
            vararg_str += f": {ast.unparse(node.args.vararg.annotation)}"
        args.append(vararg_str)

    # Handle **kwargs
    if node.args.kwarg:
        kwarg_str = f"**{node.args.kwarg.arg}"
        if node.args.kwarg.annotation:
            kwarg_str += f": {ast.unparse(node.args.kwarg.annotation)}"
        args.append(kwarg_str)

    # Build signature
    signature = f"def {node.name}({', '.join(args)})"

    # Add return type
    if node.returns:
        signature += f" -> {ast.unparse(node.returns)}"

    return signature + ":"


def extract_class_signature(node: ast.ClassDef) -> str:
    """Extract a clean class signature with inheritance."""
    bases = [ast.unparse(base) for base in node.bases]
    if bases:
        return f"class {node.name}({', '.join(bases)}):"
    return f"class {node.name}:"


def get_first_code_line(node: ast.AST, lines: List[str]) -> str:
    """Get the first non-pass line of code for context."""
    start_line = node.lineno
    for i in range(start_line, min(start_line + 3, len(lines))):
        line = lines[i].strip()
        if (
            line
            and not line.startswith("pass")
            and not line.startswith('"""')
            and not line.startswith("'''")
        ):
            return line
    return ""


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
def extract_signatures_batch() -> str:
    """Extract all function/class signatures from processed files.

    This function analyzes all processed Python files and extracts minimal
    context needed for docstring generation: signatures, type hints, and first
    line of code.
    This data is optimized for token efficiency when sent to LLM.

    Returns:
        str: JSON string containing structured signature data for all files
    """
    ensure_directories_exist()

    python_files = glob.glob(os.path.join(OUTPUT_DIR, "*.py"))
    if not python_files:
        return json.dumps({"error": "No processed Python files found"})

    all_signatures = {}

    for file_path in python_files:
        filename = os.path.basename(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                code = file.read()

            tree = ast.parse(code)
            lines = code.split("\n")
            file_signatures = {}

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    signature = extract_function_signature(node)
                    context = get_first_code_line(node, lines)
                    file_signatures[node.name] = {
                        "type": "function",
                        "signature": signature,
                        "context": context,
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                    }

                elif isinstance(node, ast.ClassDef):
                    signature = extract_class_signature(node)
                    # Get class methods for context
                    methods = []
                    for class_node in node.body:
                        if isinstance(
                            class_node,
                            (ast.FunctionDef, ast.AsyncFunctionDef),
                        ):
                            methods.append(class_node.name)

                    file_signatures[node.name] = {
                        "type": "class",
                        "signature": signature,
                        "methods": methods[:3],  # First 3 methods
                        "context": "",
                    }

            if file_signatures:
                all_signatures[filename] = file_signatures

        except Exception as e:
            all_signatures[filename] = {"error": f"Failed to parse: {str(e)}"}
    return json.dumps(all_signatures, indent=2)


# Docstring generation prompt template
DOCSTRING_GENERATION_PROMPT = """
Generate comprehensive Google-style docstrings for the following Python
functions and classes.

SIGNATURE DATA:
{signatures_data}

REQUIREMENTS:
1. Use Google-style docstring format
2. Include brief description, Args, Returns sections
3. For classes: include class description and key attributes
4. Be specific about parameter types and return types
5. Write professional, clear descriptions
6. For functions with context, infer purpose from the first line of code

RESPONSE FORMAT:
Return ONLY a JSON object with this exact structure:
{{
  "filename.py": {{
    "function_or_class_name":
        "Complete docstring content here including triple quotes",
    "another_function":
        "Another complete docstring..."
  }},
  "another_file.py": {{
    "function_name": "Docstring content..."
  }}
}}

IMPORTANT:
- Include the triple quotes in the docstring content
- Do not include any text outside the JSON structure
- Ensure all JSON is valid and properly escaped
- Generate docstrings for ALL functions and classes provided
"""


@mcp.prompt()
def generate_docstrings_prompt(signatures_data: str) -> str:
    """Generate Google-style docstrings for Python functions and classes.

    This prompt takes structured signature data and returns comprehensive
    docstrings following Google style conventions. The response is optimized
    for batch processing of multiple files.
    """
    return DOCSTRING_GENERATION_PROMPT.format(signatures_data=signatures_data)


@mcp.tool()
def apply_docstrings_batch(docstrings_json: str) -> str:
    """Apply generated docstrings to all processed files.

    This function takes the JSON response from the LLM containing docstrings
    and applies them to the appropriate functions and classes in the processed
    files. It handles proper indentation and insertion of docstrings.

    Args:
        docstrings_json (str):
            JSON string containing docstrings for each file/function

    Returns:
        str: Status report of docstring application results
    """
    ensure_directories_exist()

    try:
        docstrings_data = json.loads(docstrings_json)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON format: {str(e)}"

    if "error" in docstrings_data:
        return f"Error in docstrings data: {docstrings_data['error']}"

    results = []

    for filename, docstrings in docstrings_data.items():
        if "error" in docstrings:
            results.append(f"- {filename}: {docstrings['error']}")
            continue

        file_path = os.path.join(OUTPUT_DIR, filename)

        if not os.path.exists(file_path):
            results.append(f"- {filename}: File not found")
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                code = file.read()

            # Parse the AST to find function/class locations
            tree = ast.parse(code)
            lines = code.split("\n")

            # Apply docstrings in reverse order to maintain line numbers
            insertions = []

            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    if node.name in docstrings:
                        # Find the line after the function/class definition
                        def_line = node.lineno - 1
                        insert_line = def_line + 1

                        # Get indentation level
                        if insert_line < len(lines):
                            base_indent = len(lines[def_line]) - len(
                                lines[def_line].lstrip()
                            )
                            inner_indent = base_indent + 4
                        else:
                            inner_indent = 4

                        # Prepare docstring with proper indentation
                        docstring = docstrings[node.name]
                        docstring_lines = docstring.split("\n")
                        indented_docstring = []

                        for i, line in enumerate(docstring_lines):
                            if i == 0:
                                indented_docstring.append(
                                    " " * inner_indent + line
                                )
                            else:
                                if line.strip():
                                    indented_docstring.append(
                                        " " * inner_indent + line
                                    )
                                else:
                                    indented_docstring.append("")

                        insertions.append({
                            "line": insert_line,
                            "content": indented_docstring,
                        })

            # Sort insertions in reverse order
            insertions.sort(key=lambda x: x["line"], reverse=True)

            # Apply insertions
            for insertion in insertions:
                for i, docstring_line in enumerate(insertion["content"]):
                    lines.insert(insertion["line"] + i, docstring_line)

            # Write back to file
            updated_code = "\n".join(lines)
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(updated_code)

            applied_count = len([
                d for d in docstrings.keys() if not d.startswith("error")
            ])
            results.append(
                f"- {filename}: Applied {applied_count} docstrings"
            )

        except Exception as e:
            results.append(
                f"- {filename}: Error applying docstrings: {str(e)}"
            )

    total_files = len(docstrings_data)
    return (
        f"Docstring application completed for {total_files} files:\n"
        + "\n".join(results)
    )
