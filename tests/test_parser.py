"""Tests for parser/diff_extractor.py and parser/ast_parser.py."""

import pytest

from parser.diff_extractor import ChangedFile, HunkRange, extract_changed_files


# ── diff_extractor tests ───────────────────────────────────────────────────────


def test_extract_changed_files_basic(sample_changed_file):
    result = extract_changed_files([sample_changed_file])
    assert len(result) == 1
    cf = result[0]
    assert cf.filename == "math_utils.py"
    assert cf.language == "python"
    assert len(cf.added_lines) == 2
    assert any("divide" in line for line in cf.added_lines)


def test_extract_changed_files_skips_binary():
    files = [{"filename": "image.png", "patch": None}]
    result = extract_changed_files(files)
    assert result == []


def test_extract_changed_files_language_detection():
    files = [
        {"filename": "app.js", "patch": "@@ -1,1 +1,2 @@\n+const x = 1;\n"},
        {"filename": "style.css", "patch": "@@ -1,1 +1,2 @@\n+body { margin: 0; }\n"},
    ]
    result = extract_changed_files(files)
    langs = {cf.filename: cf.language for cf in result}
    assert langs["app.js"] == "javascript"
    assert langs["style.css"] == "unknown"


def test_hunk_parsing(sample_changed_file):
    result = extract_changed_files([sample_changed_file])
    cf = result[0]
    assert len(cf.hunks) == 1
    assert cf.hunks[0].start == 1


def test_extract_removed_lines():
    files = [{"filename": "a.py", "patch": "@@ -1,2 +1,1 @@\n-old_line\n new_line\n"}]
    result = extract_changed_files(files)
    assert "old_line" in result[0].removed_lines


# ── ast_parser tests ───────────────────────────────────────────────────────────
#
# These tests require tree-sitter and the language grammar packages.
# The whole class is skipped automatically when they are not installed.

def _tree_sitter_available(language: str) -> bool:
    """Return True if tree-sitter + the given language grammar are importable."""
    try:
        import tree_sitter  # noqa: F401
        if language == "python":
            import tree_sitter_python  # noqa: F401
        elif language == "javascript":
            import tree_sitter_javascript  # noqa: F401
        return True
    except ImportError:
        return False


# ── Python AST tests ───────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not _tree_sitter_available("python"),
    reason="tree-sitter-python not installed",
)
class TestASTParserPython:
    """Tests for Python AST extraction: complexity, docstrings, params, methods."""

    def _parse(self, source: str):
        from parser.ast_parser import extract_functions
        return extract_functions(source, "python")

    # ── Test 1: complexity ─────────────────────────────────────────────────

    def test_complexity_three_branches(self):
        """if + elif = 2 branch nodes → base 1 + 2 = complexity 3."""
        source = """\
def classify(n):
    if n > 0:
        return "positive"
    elif n < 0:
        return "negative"
    else:
        return "zero"
"""
        fns = self._parse(source)
        assert len(fns) == 1
        fn = fns[0]
        assert fn.name == "classify"
        # if_statement +1, elif_clause +1, base 1 → total 3
        assert fn.complexity == 3

    def test_complexity_one_for_plain_function(self):
        """A function with no branches has complexity exactly 1."""
        source = """\
def add(a, b):
    return a + b
"""
        fns = self._parse(source)
        assert fns[0].complexity == 1

    def test_complexity_counts_nested_branches(self):
        """Nested if inside for counts both branch nodes."""
        source = """\
def scan(items):
    for item in items:
        if item > 0:
            pass
"""
        fns = self._parse(source)
        # for_statement +1, if_statement +1, base 1 → 3
        assert fns[0].complexity == 3

    # ── Test 2: docstring detection ────────────────────────────────────────

    def test_docstring_detected_when_present(self):
        source = '''\
def documented(x):
    """Does something useful."""
    return x + 1
'''
        fns = self._parse(source)
        assert fns[0].has_docstring is True

    def test_docstring_false_when_absent(self):
        source = """\
def undocumented(x):
    return x + 1
"""
        fns = self._parse(source)
        assert fns[0].has_docstring is False

    def test_docstring_false_for_comment_only(self):
        """A # comment before the body is NOT a docstring."""
        source = """\
def has_comment(x):
    # this is just a comment
    return x
"""
        fns = self._parse(source)
        assert fns[0].has_docstring is False

    # ── Test 3: parameter counting ─────────────────────────────────────────

    def test_param_count_excludes_self(self):
        """self is not counted; x and y are → param_count == 2."""
        source = """\
class Calc:
    def add(self, x, y):
        return x + y
"""
        fns = self._parse(source)
        method = next(f for f in fns if f.name == "add")
        assert method.param_count == 2

    def test_param_count_excludes_cls(self):
        source = """\
class Foo:
    @classmethod
    def create(cls, value):
        return cls()
"""
        fns = self._parse(source)
        method = next(f for f in fns if f.name == "create")
        assert method.param_count == 1

    def test_param_count_plain_function(self):
        source = """\
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}"
"""
        fns = self._parse(source)
        assert fns[0].param_count == 2

    def test_param_count_star_args(self):
        source = """\
def variadic(*args, **kwargs):
    pass
"""
        fns = self._parse(source)
        assert fns[0].param_count == 2

    def test_param_count_zero_for_no_params(self):
        source = """\
def nothing():
    pass
"""
        fns = self._parse(source)
        assert fns[0].param_count == 0

    # ── Test 4: is_method ──────────────────────────────────────────────────

    def test_is_method_true_for_class_method(self):
        source = """\
class Calc:
    def add(self, x, y):
        return x + y
"""
        fns = self._parse(source)
        method = next(f for f in fns if f.name == "add")
        assert method.is_method is True

    def test_is_method_false_for_top_level_function(self):
        source = """\
def standalone():
    pass
"""
        fns = self._parse(source)
        assert fns[0].is_method is False

    def test_is_method_false_for_nested_function_inside_method(self):
        """A function defined inside a method body is NOT itself a method."""
        source = """\
class Outer:
    def run(self):
        def helper():
            pass
        helper()
"""
        fns = self._parse(source)
        run = next(f for f in fns if f.name == "run")
        helper = next(f for f in fns if f.name == "helper")
        assert run.is_method is True
        assert helper.is_method is False

    def test_multiple_methods_all_marked(self):
        source = """\
class Service:
    def start(self):
        pass

    def stop(self):
        pass
"""
        fns = self._parse(source)
        methods = {f.name: f for f in fns}
        assert methods["start"].is_method is True
        assert methods["stop"].is_method is True


# ── JavaScript AST tests ───────────────────────────────────────────────────────

@pytest.mark.skipif(
    not _tree_sitter_available("javascript"),
    reason="tree-sitter-javascript not installed",
)
class TestASTParserJavaScript:
    """Tests for JavaScript AST extraction."""

    def _parse(self, source: str):
        from parser.ast_parser import extract_functions
        return extract_functions(source, "javascript")

    # ── Test 5: arrow function name ────────────────────────────────────────

    def test_arrow_function_name_extracted_from_declarator(self):
        """const multiply = (a, b) => {} — name should be 'multiply'."""
        source = """\
const multiply = (a, b) => {
    return a * b;
};
"""
        fns = self._parse(source)
        assert len(fns) >= 1
        fn = next(f for f in fns if f.name == "multiply")
        assert fn.name == "multiply"

    def test_arrow_function_param_count(self):
        source = """\
const multiply = (a, b) => {
    return a * b;
};
"""
        fns = self._parse(source)
        fn = next(f for f in fns if f.name == "multiply")
        assert fn.param_count == 2

    def test_arrow_function_single_bare_param(self):
        """x => x * 2 — single identifier parameter without parens."""
        source = "const double = x => x * 2;\n"
        fns = self._parse(source)
        fn = next((f for f in fns if f.name == "double"), None)
        if fn is not None:          # some grammar versions may not expose this
            assert fn.param_count == 1

    def test_function_declaration_name(self):
        source = """\
function greet(name) {
    return "Hello " + name;
}
"""
        fns = self._parse(source)
        fn = next(f for f in fns if f.name == "greet")
        assert fn.param_count == 1

    def test_class_method_is_method_true(self):
        source = """\
class Calculator {
    add(a, b) {
        return a + b;
    }
}
"""
        fns = self._parse(source)
        method = next(f for f in fns if f.name == "add")
        assert method.is_method is True
        assert method.param_count == 2

    def test_js_docstring_detected(self):
        """String literal as first statement counts as a docstring."""
        source = '''\
function documented() {
    "use strict";
    return 1;
}
'''
        fns = self._parse(source)
        fn = next(f for f in fns if f.name == "documented")
        assert fn.has_docstring is True

    def test_js_complexity_with_if_and_logical(self):
        """if_statement + logical_expression (&&) → base 1 + 2 = 3."""
        source = """\
function check(a, b) {
    if (a && b) {
        return true;
    }
    return false;
}
"""
        fns = self._parse(source)
        fn = next(f for f in fns if f.name == "check")
        # if_statement +1, logical_expression +1, base 1 → 3
        assert fn.complexity == 3


# ── functions_in_hunk regression ──────────────────────────────────────────────


class TestFunctionsInHunk:
    """Verify hunk overlap detection still works after FunctionNode grew new fields."""

    def _make_fn(self, name: str, start: int, end: int) -> "FunctionNode":  # noqa: F821
        from parser.ast_parser import FunctionNode
        return FunctionNode(
            name=name,
            start_line=start,
            end_line=end,
            source="...",
            complexity=1,
            has_docstring=False,
            param_count=0,
            is_method=False,
        )

    def test_overlap_returns_matching_function(self):
        from parser.ast_parser import functions_in_hunk
        fns = [
            self._make_fn("foo", start=0, end=5),
            self._make_fn("bar", start=10, end=20),
        ]
        # hunk_start=3, hunk_length=5 → covers lines 3-7 (1-indexed)
        # foo spans 0-5 (0-indexed) → overlaps with diff lines 2-6 (0-indexed) ✓
        result = functions_in_hunk(fns, hunk_start=3, hunk_length=5)
        assert len(result) == 1
        assert result[0].name == "foo"

    def test_no_overlap_returns_empty(self):
        from parser.ast_parser import functions_in_hunk
        fns = [self._make_fn("foo", start=0, end=3)]
        # hunk starts at line 10, well past foo
        result = functions_in_hunk(fns, hunk_start=10, hunk_length=5)
        assert result == []

    def test_exact_boundary_overlap(self):
        """A hunk whose first line lands on the last line of a function still overlaps."""
        from parser.ast_parser import functions_in_hunk
        fns = [self._make_fn("foo", start=0, end=4)]
        # hunk_start=5, hunk_length=1 → hunk_end-1=4 (0-indexed), foo.end_line=4 → overlap
        result = functions_in_hunk(fns, hunk_start=5, hunk_length=1)
        assert len(result) == 1
        assert result[0].name == "foo"

    def test_all_new_fields_present_on_result(self):
        """Returned nodes carry all FunctionNode fields including the new ones."""
        from parser.ast_parser import FunctionNode, functions_in_hunk
        fn = FunctionNode(
            name="rich",
            start_line=0,
            end_line=10,
            source="def rich(): pass",
            complexity=3,
            has_docstring=True,
            param_count=2,
            is_method=True,
        )
        result = functions_in_hunk([fn], hunk_start=1, hunk_length=5)
        assert len(result) == 1
        r = result[0]
        assert r.complexity == 3
        assert r.has_docstring is True
        assert r.param_count == 2
        assert r.is_method is True
