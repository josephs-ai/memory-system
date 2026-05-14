"""
Tests for parse_code_cpp.py — L7-S4 of the Level 7 implementation.

Covers:
    - Basic function extraction
    - Class/struct with methods
    - Namespaces
    - Out-of-class method definitions (Foo::Bar)
    - Unreal Engine macros (UCLASS, UPROPERTY, UFUNCTION)
    - #include extraction as imports
    - Comment/docstring extraction
    - Template classes/functions
    - Chunk ID determinism
    - Schema compatibility with Python parser
    - Database integration (same upsert/stale logic)
"""

from __future__ import annotations

import os
import sys
import textwrap

import psycopg
import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from parse_code_cpp import CppParser, discover_cpp_files
from base_code_parser import BaseCodeParser

DB_DSN = os.environ.get("OPENCLAW_MEMORY_DSN", "dbname=openclaw_memory")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SIMPLE_FUNCTION = textwrap.dedent('''\
    #include <iostream>

    // Greets the user
    void sayHello(const std::string& name) {
        std::cout << "Hello, " << name << std::endl;
    }
''')

CLASS_WITH_METHODS = textwrap.dedent('''\
    #include <string>

    // A simple player class
    class Player {
    public:
        Player(const std::string& name);
        void TakeDamage(float amount);
        float GetHealth() const { return health_; }
    private:
        std::string name_;
        float health_ = 100.0f;
    };
''')

NAMESPACE_CODE = textwrap.dedent('''\
    namespace MyGame {
    namespace Core {

    void Initialize() {
        // setup
    }

    class GameState {
    public:
        void Reset() {}
    };

    }
    }
''')

OUT_OF_CLASS_METHOD = textwrap.dedent('''\
    class Calculator {
    public:
        int Add(int a, int b);
        int Multiply(int a, int b);
    };

    int Calculator::Add(int a, int b) {
        return a + b;
    }

    int Calculator::Multiply(int a, int b) {
        return a * b;
    }
''')

UNREAL_ENGINE_CODE = textwrap.dedent('''\
    #include "CoreMinimal.h"
    #include "GameFramework/Character.h"
    #include "MyCharacter.generated.h"

    UCLASS(Blueprintable, BlueprintType)
    class MYGAME_API AMyCharacter : public ACharacter {
        GENERATED_BODY()
    public:
        AMyCharacter();

        UFUNCTION(BlueprintCallable, Category = "Combat")
        void Attack(float Damage);

        UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Stats")
        float Health = 100.0f;

        UPROPERTY(VisibleAnywhere)
        float MaxHealth = 100.0f;
    };
''')

TEMPLATE_CODE = textwrap.dedent('''\
    template<typename T>
    T Max(T a, T b) {
        return (a > b) ? a : b;
    }
''')

STRUCT_CODE = textwrap.dedent('''\
    struct Vector3 {
        float x, y, z;

        float Length() const {
            return sqrt(x*x + y*y + z*z);
        }
    };
''')

EMPTY_FILE = ""


# ---------------------------------------------------------------------------
# Tests: BaseCodeParser interface
# ---------------------------------------------------------------------------

class TestBaseInterface:
    def test_cpp_parser_is_base_parser(self):
        p = CppParser()
        assert isinstance(p, BaseCodeParser)

    def test_language_id(self):
        p = CppParser()
        assert p.language_id == "cpp"

    def test_file_extensions(self):
        p = CppParser()
        assert ".cpp" in p.file_extensions
        assert ".h" in p.file_extensions
        assert ".hpp" in p.file_extensions

    def test_can_parse(self):
        p = CppParser()
        assert p.can_parse("foo.cpp")
        assert p.can_parse("bar.h")
        assert not p.can_parse("baz.py")

    def test_chunk_id_matches_python_parser(self):
        """Chunk IDs use the same algorithm as the Python parser."""
        from parse_code_ast import make_chunk_id
        py_id = make_chunk_id("/a/b.cpp", "function", "b.foo")
        cpp_id = BaseCodeParser.make_chunk_id("/a/b.cpp", "function", "b.foo")
        assert py_id == cpp_id


# ---------------------------------------------------------------------------
# Tests: Parse File
# ---------------------------------------------------------------------------

class TestCppParsing:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.parser = CppParser()

    def test_simple_function(self):
        chunks = self.parser.parse_file("/test/hello.cpp", SIMPLE_FUNCTION)
        fns = [c for c in chunks if c["chunk_type"] == "function"]
        assert len(fns) == 1
        assert fns[0]["qualified_name"] == "hello.sayHello"
        assert "void sayHello" in fns[0]["signature"]
        assert fns[0]["docstring"] is not None
        assert "Greets" in fns[0]["docstring"]

    def test_includes_as_imports(self):
        chunks = self.parser.parse_file("/test/hello.cpp", SIMPLE_FUNCTION)
        mod = [c for c in chunks if c["chunk_type"] == "module"][0]
        assert "iostream" in mod["imports"]

    def test_class_with_methods(self):
        chunks = self.parser.parse_file("/test/player.h", CLASS_WITH_METHODS)
        classes = [c for c in chunks if c["chunk_type"] == "class"]
        assert len(classes) == 1
        assert classes[0]["qualified_name"] == "player.Player"
        assert "class Player" in classes[0]["signature"]

        methods = [c for c in chunks if c["chunk_type"] == "method"]
        # GetHealth has an inline body
        assert any(m["qualified_name"] == "player.Player.GetHealth" for m in methods)

    def test_namespace(self):
        chunks = self.parser.parse_file("/test/game.cpp", NAMESPACE_CODE)
        fns = [c for c in chunks if c["chunk_type"] == "function"]
        assert any("Initialize" in f["qualified_name"] for f in fns)

        classes = [c for c in chunks if c["chunk_type"] == "class"]
        assert any("GameState" in c["qualified_name"] for c in classes)

    def test_out_of_class_method(self):
        chunks = self.parser.parse_file("/test/calc.cpp", OUT_OF_CLASS_METHOD)
        methods = [c for c in chunks if c["chunk_type"] == "method"]
        method_names = {m["qualified_name"] for m in methods}
        assert "calc.Calculator.Add" in method_names
        assert "calc.Calculator.Multiply" in method_names

        for m in methods:
            assert m["class_name"] == "Calculator"

    def test_unreal_engine_macros(self):
        chunks = self.parser.parse_file("/test/MyCharacter.h", UNREAL_ENGINE_CODE)
        classes = [c for c in chunks if c["chunk_type"] == "class"]
        assert len(classes) == 1

        cls = classes[0]
        assert "AMyCharacter" in cls["qualified_name"]
        assert "ACharacter" in cls["signature"]

        # Check includes
        mod = [c for c in chunks if c["chunk_type"] == "module"][0]
        assert "CoreMinimal.h" in mod["imports"]
        assert "MyCharacter.generated.h" in mod["imports"]

    def test_struct(self):
        chunks = self.parser.parse_file("/test/vec.h", STRUCT_CODE)
        classes = [c for c in chunks if c["chunk_type"] == "class"]
        assert len(classes) == 1
        assert "struct Vector3" in classes[0]["signature"]

        methods = [c for c in chunks if c["chunk_type"] == "method"]
        assert any("Length" in m["qualified_name"] for m in methods)

    def test_template(self):
        chunks = self.parser.parse_file("/test/tmpl.h", TEMPLATE_CODE)
        fns = [c for c in chunks if c["chunk_type"] == "function"]
        assert len(fns) == 1
        assert "Max" in fns[0]["qualified_name"]
        assert fns[0]["decorators"] == ["template"]

    def test_empty_file(self):
        chunks = self.parser.parse_file("/test/empty.cpp", EMPTY_FILE)
        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "module"

    def test_line_numbers(self):
        chunks = self.parser.parse_file("/test/hello.cpp", SIMPLE_FUNCTION)
        for chunk in chunks:
            assert chunk["line_start"] >= 1
            assert chunk["line_end"] >= chunk["line_start"]

    def test_file_hash_consistent(self):
        chunks = self.parser.parse_file("/test/hello.cpp", SIMPLE_FUNCTION)
        hashes = {c["file_hash"] for c in chunks}
        assert len(hashes) == 1

    def test_schema_compatibility(self):
        """All chunks have the same keys as Python parser output."""
        required_keys = {
            "id", "file_path", "file_hash", "chunk_type", "qualified_name",
            "module_name", "signature", "docstring", "body", "decorators",
            "imports", "line_start", "line_end", "parent_chunk_id", "class_name",
        }
        chunks = self.parser.parse_file("/test/hello.cpp", SIMPLE_FUNCTION)
        for chunk in chunks:
            assert required_keys.issubset(set(chunk.keys())), \
                f"Missing keys: {required_keys - set(chunk.keys())}"


# ---------------------------------------------------------------------------
# Tests: File Discovery
# ---------------------------------------------------------------------------

class TestCppFileDiscovery:
    def test_single_file(self, tmp_path):
        f = tmp_path / "test.cpp"
        f.write_text("int main() {}")
        result = discover_cpp_files(str(f))
        assert len(result) == 1

    def test_directory(self, tmp_path):
        (tmp_path / "a.cpp").write_text("void a() {}")
        (tmp_path / "b.h").write_text("class B {};")
        result = discover_cpp_files(str(tmp_path))
        assert len(result) == 2

    def test_skips_intermediate(self, tmp_path):
        (tmp_path / "a.cpp").write_text("void a() {}")
        inter = tmp_path / "Intermediate"
        inter.mkdir()
        (inter / "b.cpp").write_text("void b() {}")
        result = discover_cpp_files(str(tmp_path))
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Tests: Database Integration
# ---------------------------------------------------------------------------

@pytest.fixture
def db_conn():
    conn = psycopg.connect(DB_DSN)
    yield conn
    with conn.cursor() as cur:
        cur.execute("DELETE FROM code_chunks WHERE file_path LIKE '/test/%'")
    conn.commit()
    conn.close()


class TestCppDatabaseOps:
    def test_upsert_cpp_chunks(self, db_conn):
        from parse_code_ast import upsert_chunks
        parser = CppParser()
        chunks = parser.parse_file("/test/hello.cpp", SIMPLE_FUNCTION)
        count = upsert_chunks(db_conn, chunks)
        assert count == len(chunks)

        with db_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM code_chunks WHERE file_path = '/test/hello.cpp'")
            assert cur.fetchone()[0] == len(chunks)

    def test_cpp_and_python_coexist(self, db_conn):
        """C++ and Python chunks share the same table without conflicts."""
        from parse_code_ast import upsert_chunks, parse_file

        py_chunks = parse_file("/test/example.py", "def foo(): pass\n")
        upsert_chunks(db_conn, py_chunks)

        cpp_parser = CppParser()
        cpp_chunks = cpp_parser.parse_file("/test/example.cpp", SIMPLE_FUNCTION)
        upsert_chunks(db_conn, cpp_chunks)

        with db_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM code_chunks WHERE file_path LIKE '/test/%'")
            total = cur.fetchone()[0]
        assert total == len(py_chunks) + len(cpp_chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
