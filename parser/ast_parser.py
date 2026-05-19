"""tree-sitter based AST parser for Python and JavaScript.

Extracts FunctionNode objects enriched with:
  - Cyclomatic complexity  (branch node count + 1 base)
  - Docstring presence     (first statement is a string literal)
  - Parameter count        (self/cls excluded for Python)
  - is_method flag         (True when defined directly inside a class body)

Parsers are loaded lazily and cached at module level so repeated calls to
extract_functions() pay the import cost only once per language.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Module-level parser cache: language -> tree_sitter.Parser
_PARSERS: dict[str, object] = {}


# ── Node type tables ───────────────────────────────────────────────────────────

# Function-like node types for each language.
_FUNCTION_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"function_definition"}),
    "javascript": frozenset({
        "function_declaration",   # function foo() {}
        "function",               # const foo = function() {}
        "arrow_function",         # const foo = () => {}
        "method_definition",      # class Foo { bar() {} }
    }),
}

# Class node types — entering one increments class_depth in the walk.
_CLASS_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"class_definition"}),
    "javascript": frozenset({"class_declaration", "class"}),
}

# Branch node types counted towards cyclomatic complexity.
_COMPLEXITY_NODES: dict[str, frozenset[str]] = {
    "python": frozenset({
        "if_statement",
        "elif_clause",
        "for_statement",
        "while_statement",
        "try_statement",
        "except_clause",
        "with_statement",
        "boolean_operator",        # 'and' / 'or'
    }),
    "javascript": frozenset({
        "if_statement",
        "else_if_clause",          # kept for grammar completeness
        "for_statement",
        "for_in_statement",
        "while_statement",
        "try_statement",
        "catch_clause",
        # NOTE: tree-sitter-javascript has no 'logical_expression' node.
        # Logical operators (&&, ||) are represented as binary_expression
        # nodes with an anonymous operator token.  Handled separately in
        # _compute_complexity() via _is_logical_binary().
    }),
}

# Node type of the function body container (used for docstring detection).
_BODY_TYPES: dict[str, str] = {
    "python": "block",
    "javascript": "statement_block",
}

# String literal node types per language.
_STRING_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"string"}),
    "javascript": frozenset({"string", "template_string"}),
}

# Node type of the formal parameter list.
_PARAM_CONTAINER: dict[str, str] = {
    "python": "parameters",
    "javascript": "formal_parameters",
}

# Named parameter node types for Python (each counts as one parameter).
_PY_PARAM_TYPES: frozenset[str] = frozenset({
    "identifier",
    "default_parameter",
    "typed_parameter",
    "typed_default_parameter",
    "list_splat_pattern",         # *args
    "dictionary_splat_pattern",   # **kwargs
})


# ── Public data classes ────────────────────────────────────────────────────────


@dataclass
class FunctionNode:
    """All statically-extractable facts about one function or method."""

    name: str
    start_line: int               # 0-indexed (matches tree-sitter row)
    end_line: int                 # 0-indexed, inclusive
    source: str                   # raw source text of the whole function
    complexity: int = 1           # cyclomatic complexity (≥1)
    has_docstring: bool = False
    param_count: int = 0          # excludes self/cls for Python
    is_method: bool = False       # True if defined directly inside a class


# ── Parser loading ─────────────────────────────────────────────────────────────


def _get_parser(language: str):
    """Return a cached tree-sitter Parser, or None on missing dependency.

    Args:
        language: "python" or "javascript".

    Returns:
        tree_sitter.Parser instance, or None if the grammar isn't installed.
    """
    if language in _PARSERS:
        return _PARSERS[language]

    try:
        from tree_sitter import Language, Parser

        if language == "python":
            import tree_sitter_python as tspython
            lang = Language(tspython.language())
        elif language == "javascript":
            import tree_sitter_javascript as tsjavascript
            lang = Language(tsjavascript.language())
        else:
            return None

        parser = Parser(lang)
        _PARSERS[language] = parser
        return parser
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not load tree-sitter parser for %s: %s", language, exc)
        return None


# ── Main extraction entry point ────────────────────────────────────────────────


def extract_functions(source_code: str, language: str) -> list[FunctionNode]:
    """Parse source_code and return all function/method definitions.

    Handles top-level functions, class methods (is_method=True), nested
    functions, and JavaScript arrow functions assigned to variables.

    Args:
        source_code: Complete source text to parse.
        language:    "python" or "javascript".

    Returns:
        List of FunctionNode, one per definition found.  Empty list if the
        parser is unavailable or the source fails to parse.
    """
    parser = _get_parser(language)
    if parser is None:
        return []

    tree = parser.parse(source_code.encode())
    lines = source_code.splitlines()
    functions: list[FunctionNode] = []

    func_types = _FUNCTION_TYPES.get(language, frozenset())
    class_types = _CLASS_TYPES.get(language, frozenset())

    def _walk(node, parent=None, class_depth: int = 0, func_depth: int = 0) -> None:
        """Recursive pre-order walk.

        Args:
            node:        Current tree-sitter Node.
            parent:      Parent Node (needed for arrow-function name lookup).
            class_depth: How many class scopes deep we are.
            func_depth:  How many function scopes deep we are.
        """
        is_func = node.type in func_types
        is_class = node.type in class_types

        if is_func:
            start = node.start_point[0]
            end = node.end_point[0]
            source = "\n".join(lines[start: end + 1])

            functions.append(FunctionNode(
                name=_get_function_name(node, language, parent),
                start_line=start,
                end_line=end,
                source=source,
                complexity=_compute_complexity(node, language),
                has_docstring=_has_docstring(node, language),
                param_count=_count_params(node, language),
                # A function is a *method* only when it sits directly inside a
                # class scope and not already inside another function scope.
                is_method=class_depth > 0 and func_depth == 0,
            ))
            # Recurse into the function body with incremented func_depth so
            # nested functions are not marked as methods.
            for child in node.children:
                _walk(child, parent=node,
                      class_depth=class_depth,
                      func_depth=func_depth + 1)
            return  # children already visited above

        new_class_depth = class_depth + (1 if is_class else 0)
        for child in node.children:
            _walk(child, parent=node,
                  class_depth=new_class_depth,
                  func_depth=func_depth)

    _walk(tree.root_node)
    return functions


# ── Name extraction ────────────────────────────────────────────────────────────


def _get_function_name(node, language: str, parent=None) -> str:
    """Return the declared name of a function node.

    Special cases handled:
    - JS arrow_function / function expression: name lives in parent
      variable_declarator as an identifier sibling.
    - JS method_definition: name is a property_identifier child.
    - Everything else: first identifier child.

    Returns "<anonymous>" when no name can be found.
    """
    # ── JavaScript: arrow function / function expression ──────────────────
    if language == "javascript" and node.type in ("arrow_function", "function"):
        if parent is not None and parent.type == "variable_declarator":
            for child in parent.children:
                if child.type == "identifier":
                    return child.text.decode()

    # ── JavaScript: method_definition uses property_identifier ───────────
    if node.type == "method_definition":
        for child in node.children:
            if child.type in ("property_identifier", "identifier"):
                return child.text.decode()

    # ── Default: first identifier child ──────────────────────────────────
    for child in node.children:
        if child.type == "identifier":
            return child.text.decode()

    return "<anonymous>"


# ── Complexity scoring ─────────────────────────────────────────────────────────


def _compute_complexity(func_node, language: str) -> int:
    """Cyclomatic complexity = 1 + number of branch nodes in the function.

    Walks all descendants of func_node, counting every node whose type
    appears in _COMPLEXITY_NODES for the given language.  Nested function
    definitions are included in the count (their branches are reachable from
    the outer function).

    JavaScript special case: tree-sitter-javascript has no 'logical_expression'
    node.  Logical operators (&&, ||) appear as binary_expression nodes with an
    anonymous operator token child.  These are detected via _is_logical_binary()
    and counted as branch points because they represent conditional short-circuit
    evaluation paths.

    Args:
        func_node: The function definition tree-sitter Node.
        language:  "python" or "javascript".

    Returns:
        Integer ≥ 1.
    """
    branch_types = _COMPLEXITY_NODES.get(language, frozenset())
    count = 1  # base complexity

    def _walk(node) -> None:
        nonlocal count
        for child in node.children:
            if child.type in branch_types:
                count += 1
            elif language == "javascript" and child.type == "binary_expression":
                if _is_logical_binary(child):
                    count += 1
            _walk(child)

    _walk(func_node)
    return count


def _is_logical_binary(node) -> bool:
    """Return True if a binary_expression node uses && or || as its operator.

    In tree-sitter-javascript, the operator is an anonymous token child
    sandwiched between the two named operand children.
    """
    for child in node.children:
        if not child.is_named and child.type in ("&&", "||"):
            return True
    return False


# ── Docstring detection ────────────────────────────────────────────────────────


def _has_docstring(func_node, language: str) -> bool:
    """Return True if the function's first statement is a string literal.

    Python:     function body is a `block`; first named child must be an
                `expression_statement` containing a `string` node.
    JavaScript: function body is a `statement_block`; same pattern with
                `string` or `template_string`.

    Args:
        func_node: The function definition Node.
        language:  "python" or "javascript".

    Returns:
        True if a docstring is detected, False otherwise.
    """
    body_type = _BODY_TYPES.get(language)
    string_types = _STRING_TYPES.get(language, frozenset())

    for child in func_node.children:
        if child.type != body_type:
            continue
        # Iterate the body's children, skipping anonymous tokens ({, }, \n).
        for stmt in child.children:
            if not stmt.is_named:
                continue  # punctuation / whitespace tokens
            if stmt.type == "expression_statement":
                # The expression_statement's child must be a string literal.
                for sub in stmt.children:
                    if sub.type in string_types:
                        return True
                # An expression statement exists but is not a string.
                return False
            else:
                # First named statement is not an expression_statement.
                return False

    return False


# ── Parameter counting ─────────────────────────────────────────────────────────


def _count_params(func_node, language: str) -> int:
    """Count the declared parameters, excluding Python's self/cls.

    Python parameter node types handled:
        identifier, default_parameter, typed_parameter,
        typed_default_parameter, list_splat_pattern, dictionary_splat_pattern

    JavaScript:
        Counts named children of formal_parameters.
        Arrow functions with a single bare identifier (x => ...) are handled
        by inspecting the "parameter" field directly.

    Args:
        func_node: Function definition Node.
        language:  "python" or "javascript".

    Returns:
        Non-negative integer.
    """
    if language == "python":
        return _count_params_python(func_node)
    if language == "javascript":
        return _count_params_javascript(func_node)
    return 0


def _count_params_python(func_node) -> int:
    """Count Python parameters, excluding self and cls."""
    for child in func_node.children:
        if child.type != "parameters":
            continue
        count = 0
        for param in child.children:
            if not param.is_named:
                continue
            if param.type not in _PY_PARAM_TYPES:
                continue
            name = _extract_py_param_name(param)
            if name not in ("self", "cls", ""):
                count += 1
        return count
    return 0


def _extract_py_param_name(param_node) -> str:
    """Get the bare identifier name from any Python parameter node type."""
    if param_node.type == "identifier":
        return param_node.text.decode()
    # default_parameter, typed_parameter, typed_default_parameter, *args, **kwargs
    # all have the parameter name as their first identifier descendant.
    for child in param_node.children:
        if child.type == "identifier":
            return child.text.decode()
    return ""


def _count_params_javascript(func_node) -> int:
    """Count JavaScript parameters from formal_parameters or bare arrow param."""
    # Arrow function with a single bare parameter: x => expr
    # tree-sitter exposes this via the "parameter" field as a plain identifier.
    if func_node.type == "arrow_function":
        try:
            param_field = func_node.child_by_field_name("parameter")
            if param_field is not None and param_field.type == "identifier":
                return 1
        except Exception:
            pass  # child_by_field_name not available in this version

    # All other cases: formal_parameters node holds the parameter list.
    for child in func_node.children:
        if child.type == "formal_parameters":
            # Count only named children (skips '(', ',', ')' anonymous tokens).
            return sum(1 for c in child.children if c.is_named)

    return 0


# ── Hunk overlap ───────────────────────────────────────────────────────────────


def functions_in_hunk(
    functions: list[FunctionNode],
    hunk_start: int,
    hunk_length: int,
) -> list[FunctionNode]:
    """Return every FunctionNode whose body overlaps with a diff hunk.

    Args:
        functions:   All parsed FunctionNodes for a file.
        hunk_start:  First line of the hunk, 1-indexed (unified diff format).
        hunk_length: Number of lines in the hunk.

    Returns:
        Subset of functions that overlap with the hunk range.
    """
    hunk_end = hunk_start + hunk_length - 1
    # Unified-diff lines are 1-indexed; FunctionNode lines are 0-indexed.
    return [
        f
        for f in functions
        if f.start_line <= hunk_end - 1 and f.end_line >= hunk_start - 1
    ]
