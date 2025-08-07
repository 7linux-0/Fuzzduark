"""
Enhanced Python source code mutator using Hypothesis strategies and Lark grammar.
Supports AST subtree manipulation with code-aware mutations.
"""

import ast
import sys
import random
import copy
from typing import Optional, List, Set, Union, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from hypothesis import strategies as st, find, assume
from hypothesis.extra.lark import LarkStrategy
from lark import Lark
from lark.indenter import Indenter

# --- Configuration and Setup ---

# The LARK_GRAMMAR is now loaded from a local file.
try:
    with open("python.lark", "r", encoding="utf8") as f:
        LARK_GRAMMAR = f.read()
except FileNotFoundError:
    print(
        "ERROR: 'python.lark' not found. Please download it by running:\n"
        "wget https://raw.githubusercontent.com/lark-parser/lark/master/lark/grammars/python.lark"
    )
    sys.exit(1)


# --- Mutation Strategy Types ---

class MutationStrategy(Enum):
    """Different mutation strategies available"""
    GRAMMAR_BASED = "grammar"  # Original grammar-based generation
    COMPREHENSION_FOCUSED = "comprehension"  # Focus on list comprehensions
    LAMBDA_FOCUSED = "lambda"  # Focus on lambda expressions
    SIMPLE_IDENTIFIERS = "simple"  # Use only simple identifiers like __a, __b
    SUBTREE_MANIPULATION = "subtree"  # Add/remove/replace subtrees
    CODE_AWARE = "codeaware"  # Use subtrees from the original code


class MutationOperation(Enum):
    """Types of mutations for subtree manipulation"""
    REPLACE = "replace"  # Replace a node with generated one
    ADD = "add"  # Add a new node (e.g., add statement to function body)
    DELETE = "delete"  # Remove a node from the tree
    SWAP = "swap"  # Swap two compatible nodes
    REUSE = "reuse"  # Replace with subtree from elsewhere in the code


@dataclass
class MutationConfig:
    """Configuration for mutation behavior"""
    strategy: MutationStrategy = MutationStrategy.GRAMMAR_BASED
    operation: MutationOperation = MutationOperation.REPLACE
    num_mutations: int = 1
    seed: Optional[int] = None
    allowed_identifiers: Optional[Set[str]] = None
    filtered_keywords: Optional[Set[str]] = None
    max_complexity: int = 3
    preserve_signatures: bool = True  # Preserve function/class signatures


# --- Core Grammar Setup ---

COMPILE_MODES = {
    "eval_input": "eval",
    "file_input": "exec",
    "stmt": "single",
    "simple_stmt": "single",
    "compound_stmt": "single",
}
ALLOWED_CHARS = st.characters(codec="utf-8", min_codepoint=1)


class PythonIndenter(Indenter):
    NL_type = "_NEWLINE"
    OPEN_PAREN_types = ["LPAR", "LSQB", "LBRACE"]
    CLOSE_PAREN_types = ["RPAR", "RSQB", "RBRACE"]
    INDENT_type = "_INDENT"
    DEDENT_type = "_DEDENT"
    tab_len = 4


# --- AST Helpers ---

def create_empty_arguments() -> ast.arguments:
    """Create a properly initialized ast.arguments object"""
    return ast.arguments(
        posonlyargs=[],
        args=[],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[]
    )


def fix_missing_locations(node: ast.AST, lineno: int = 1, col_offset: int = 0) -> ast.AST:
    """Recursively fix missing location information in AST nodes"""
    for child in ast.walk(node):
        if not hasattr(child, 'lineno'):
            child.lineno = lineno
        if not hasattr(child, 'col_offset'):
            child.col_offset = col_offset
    return node


def get_node_size(node: ast.AST) -> int:
    """Get the size (number of subnodes) of an AST node"""
    return len(list(ast.walk(node)))


def is_meaningful_node(node: ast.AST) -> bool:
    """Check if a node is meaningful for mutation (not just context or simple names)"""
    # Skip context nodes and single names
    if isinstance(node, (ast.Load, ast.Store, ast.Del)):
        return False
    if isinstance(node, ast.Name) and not hasattr(node, 'parent'):
        return False
    # Skip docstrings
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        parent = getattr(node, 'parent', None)
        if parent and isinstance(parent, (ast.FunctionDef, ast.ClassDef)):
            if parent.body and parent.body[0] is node:
                return False
    return True


# --- Subtree Collection and Analysis ---

class SubtreeCollector(ast.NodeVisitor):
    """Collects reusable subtrees from the AST"""
    
    def __init__(self):
        self.expressions = []
        self.statements = []
        self.functions = []
        self.classes = []
        self.comprehensions = []
        self.lambdas = []
        self._parent_map = {}
    
    def collect(self, tree: ast.AST) -> None:
        """Collect all reusable subtrees from the AST"""
        # Build parent map
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                self._parent_map[child] = parent
                setattr(child, 'parent', parent)
        
        # Collect nodes
        self.visit(tree)
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.functions.append(node)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef):
        self.classes.append(node)
        self.generic_visit(node)
    
    def visit_Lambda(self, node: ast.Lambda):
        self.lambdas.append(node)
        self.generic_visit(node)
    
    def visit_ListComp(self, node: ast.ListComp):
        self.comprehensions.append(node)
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign):
        self.statements.append(node)
        self.generic_visit(node)
    
    def visit_Expr(self, node: ast.Expr):
        if is_meaningful_node(node.value):
            self.expressions.append(node.value)
        self.statements.append(node)
        self.generic_visit(node)
    
    def visit_If(self, node: ast.If):
        self.statements.append(node)
        self.generic_visit(node)
    
    def visit_For(self, node: ast.For):
        self.statements.append(node)
        self.generic_visit(node)
    
    def visit_While(self, node: ast.While):
        self.statements.append(node)
        self.generic_visit(node)
    
    def visit_Return(self, node: ast.Return):
        self.statements.append(node)
        self.generic_visit(node)
    
    def get_compatible_subtrees(self, target_node: ast.AST) -> List[ast.AST]:
        """Get subtrees that can replace the target node"""
        compatible = []
        
        if isinstance(target_node, ast.stmt):
            compatible.extend(self.statements)
        elif isinstance(target_node, ast.expr):
            compatible.extend(self.expressions)
        elif isinstance(target_node, ast.FunctionDef):
            compatible.extend(self.functions)
        elif isinstance(target_node, ast.ClassDef):
            compatible.extend(self.classes)
        elif isinstance(target_node, ast.Lambda):
            compatible.extend(self.lambdas)
        elif isinstance(target_node, ast.ListComp):
            compatible.extend(self.comprehensions)
        
        # Filter out the target node itself
        return [n for n in compatible if n is not target_node]


# --- Custom Strategies ---

def create_custom_strategies(config: MutationConfig):
    """Create custom strategies based on configuration"""
    
    if config.allowed_identifiers:
        identifiers = st.sampled_from(list(config.allowed_identifiers))
    else:
        identifiers = st.one_of(
            st.just("__a"),
            st.just("__b"),
            st.text(st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=5).filter(str.isidentifier)
        )
    
    if config.filtered_keywords:
        identifiers = identifiers.filter(lambda x: x not in config.filtered_keywords)
    
    return identifiers


def simple_value_expr():
    """Generate simple value expressions"""
    return st.one_of(
        st.builds(ast.Constant, st.integers(min_value=1, max_value=10)),
        st.builds(ast.Name, st.just("__a"), st.just(ast.Load())),
        st.builds(ast.Name, st.just("__b"), st.just(ast.Load())),
    )


@st.composite
def simple_lambda(draw, preserve_args=None):
    """Generate lambda expressions, optionally preserving arguments"""
    body = draw(simple_value_expr())
    
    if preserve_args:
        args = copy.deepcopy(preserve_args)
    else:
        args = create_empty_arguments()
        args.args = [ast.arg("x", annotation=None) for _ in range(draw(st.integers(0, 2)))]
    
    return ast.Lambda(args, body)


@st.composite
def simple_listcomp(draw):
    """Generate simple list comprehensions"""
    elt = draw(simple_value_expr())
    target = ast.Name("i", ast.Store())
    iter_expr = ast.Name("__a", ast.Load())
    comp = ast.comprehension(target, iter_expr, [], is_async=0)
    return ast.ListComp(elt, [comp])


@st.composite
def simple_statement(draw):
    """Generate simple statements"""
    stmt_type = draw(st.sampled_from(['assign', 'expr', 'pass', 'return']))
    
    if stmt_type == 'assign':
        target = ast.Name(draw(st.sampled_from(["__a", "__b", "x"])), ast.Store())
        value = draw(simple_value_expr())
        return ast.Assign([target], value)
    elif stmt_type == 'expr':
        return ast.Expr(draw(simple_value_expr()))
    elif stmt_type == 'return':
        return ast.Return(draw(simple_value_expr()))
    else:
        return ast.Pass()


# --- Enhanced Grammar Strategy ---

class EnhancedGrammarStrategy(LarkStrategy):
    def __init__(self, grammar: Lark, start: str, config: MutationConfig):
        self.config = config
        
        explicit_strategies = {
            PythonIndenter.INDENT_type: st.just(" " * PythonIndenter.tab_len),
            PythonIndenter.DEDENT_type: st.just(""),
        }
        
        if config.strategy == MutationStrategy.SIMPLE_IDENTIFIERS:
            explicit_strategies["NAME"] = st.sampled_from(["__a", "__b", "x", "y"])
        else:
            name_strategy = st.text(
                st.characters(min_codepoint=97, max_codepoint=122), 
                min_size=1,
                max_size=5
            ).filter(str.isidentifier)
            
            if config.filtered_keywords:
                name_strategy = name_strategy.filter(
                    lambda x: x not in config.filtered_keywords
                )
            
            explicit_strategies["NAME"] = name_strategy
        
        super().__init__(grammar, start, explicit_strategies, alphabet=ALLOWED_CHARS)

    def do_draw(self, data):
        result = super().do_draw(data)
        try:
            ast.parse(result)
        except Exception:
            assume(False)
        return result


# --- Enhanced Code Mutator ---

class EnhancedCodeMutator(ast.NodeTransformer):
    """Enhanced AST transformer supporting multiple mutations and custom strategies"""

    def __init__(self, config: MutationConfig):
        self.config = config
        self._mutations_done = 0
        self._parent_map = {}
        self._subtree_collector = SubtreeCollector()
        
        if config.seed is not None:
            random.seed(config.seed)

    def build_parent_map(self, node: ast.AST, parent: Optional[ast.AST] = None):
        """Build a map of nodes to their parents"""
        self._parent_map[node] = parent
        setattr(node, 'parent', parent)
        for child in ast.iter_child_nodes(node):
            self.build_parent_map(child, node)

    def perform_mutations(self, root: ast.AST) -> ast.AST:
        """Perform configured mutations on the AST"""
        self.build_parent_map(root)
        
        # Collect subtrees for code-aware mutations
        if self.config.strategy in [MutationStrategy.CODE_AWARE, MutationStrategy.SUBTREE_MANIPULATION]:
            self._subtree_collector.collect(root)
        
        if self.config.strategy == MutationStrategy.SUBTREE_MANIPULATION:
            return self.perform_subtree_mutations(root)
        else:
            return self.perform_replacement_mutations(root)

    def perform_subtree_mutations(self, root: ast.AST) -> ast.AST:
        """Perform add/delete/swap operations on the AST"""
        for i in range(self.config.num_mutations):
            if self.config.operation == MutationOperation.DELETE:
                root = self.delete_meaningful_node(root)
            elif self.config.operation == MutationOperation.ADD:
                root = self.add_random_node(root)
            elif self.config.operation == MutationOperation.SWAP:
                root = self.swap_random_nodes(root)
            elif self.config.operation == MutationOperation.REUSE:
                root = self.reuse_subtree(root)
            else:  # REPLACE
                root = self.perform_replacement_mutations(root, single=True)
        
        return root

    def delete_meaningful_node(self, root: ast.AST) -> ast.AST:
        """Delete a meaningful node from the AST"""
        deletable_nodes = []
        
        for node in ast.walk(root):
            parent = self._parent_map.get(node)
            if not parent:
                continue
                
            # Only consider meaningful nodes for deletion
            if not is_meaningful_node(node):
                continue
                
            # Check if it's safe to delete
            if isinstance(parent, (ast.FunctionDef, ast.ClassDef, ast.If, ast.For, ast.While)):
                if hasattr(parent, 'body') and len(parent.body) > 1 and node in parent.body:
                    deletable_nodes.append(node)
            elif isinstance(parent, ast.Module) and len(parent.body) > 1:
                deletable_nodes.append(node)
        
        if not deletable_nodes:
            print("No safely deletable meaningful nodes found")
            return root
        
        node_to_delete = random.choice(deletable_nodes)
        parent = self._parent_map[node_to_delete]
        
        print(f"\n--- Deleting {type(node_to_delete).__name__} statement ---")
        if hasattr(node_to_delete, 'name'):
            print(f"    (name: {node_to_delete.name})")
        
        # Remove the node from its parent
        for attr_name in parent._fields:
            attr_value = getattr(parent, attr_name, None)
            if isinstance(attr_value, list) and node_to_delete in attr_value:
                attr_value.remove(node_to_delete)
                if not attr_value and attr_name == 'body':
                    attr_value.append(ast.Pass())
        
        return root

    def add_random_node(self, root: ast.AST) -> ast.AST:
        """Add a random node to the AST"""
        addable_nodes = []
        
        for node in ast.walk(root):
            if hasattr(node, 'body') and isinstance(node.body, list):
                addable_nodes.append(node)
        
        if not addable_nodes:
            print("No nodes found where we can add children")
            return root
        
        parent = random.choice(addable_nodes)
        
        # If code-aware, try to reuse an existing statement
        if self._subtree_collector.statements:
            original_stmt = random.choice(self._subtree_collector.statements)
            new_stmt = copy.deepcopy(original_stmt)
            print(f"\n--- Adding {type(new_stmt).__name__} (reused) to {type(parent).__name__} ---")
        else:
            new_stmt = find(simple_statement(), lambda x: True)
            print(f"\n--- Adding {type(new_stmt).__name__} (generated) to {type(parent).__name__} ---")
        
        if hasattr(parent, 'lineno'):
            fix_missing_locations(new_stmt, parent.lineno, parent.col_offset)
        
        position = random.randint(0, len(parent.body))
        parent.body.insert(position, new_stmt)
        
        return root

    def swap_random_nodes(self, root: ast.AST) -> ast.AST:
        """Swap two random compatible nodes"""
        # Group nodes by type
        nodes_by_type = defaultdict(list)
        
        for node in ast.walk(root):
            if is_meaningful_node(node):
                nodes_by_type[type(node).__name__].append(node)
        
        # Find swappable pairs
        swappable_types = [t for t, nodes in nodes_by_type.items() if len(nodes) >= 2]
        
        if not swappable_types:
            print("No swappable pairs found")
            return root
        
        chosen_type = random.choice(swappable_types)
        nodes = nodes_by_type[chosen_type]
        node1, node2 = random.sample(nodes, 2)
        
        print(f"\n--- Swapping two {chosen_type} nodes ---")
        
        # Swap their parents' references
        parent1 = self._parent_map.get(node1)
        parent2 = self._parent_map.get(node2)
        
        if parent1 and parent2:
            # Find and swap in parent attributes
            for p, n, other in [(parent1, node1, node2), (parent2, node2, node1)]:
                for attr in p._fields:
                    val = getattr(p, attr, None)
                    if isinstance(val, list):
                        try:
                            idx = val.index(n)
                            val[idx] = other
                        except ValueError:
                            pass
                    elif val is n:
                        setattr(p, attr, other)
        
        return root

    def reuse_subtree(self, root: ast.AST) -> ast.AST:
        """Replace a node with a compatible subtree from elsewhere in the code"""
        replaceable_nodes = [n for n in ast.walk(root) if is_meaningful_node(n)]
        
        if not replaceable_nodes:
            print("No replaceable nodes found")
            return root
        
        target = random.choice(replaceable_nodes)
        compatible = self._subtree_collector.get_compatible_subtrees(target)
        
        if not compatible:
            print(f"No compatible subtrees found for {type(target).__name__}")
            return root
        
        replacement = random.choice(compatible)
        print(f"\n--- Replacing {type(target).__name__} with reused {type(replacement).__name__} ---")
        
        replacer = NodeReplacer(self.config, {id(target)}, replacement)
        return replacer.visit(root)

    def perform_replacement_mutations(self, root: ast.AST, single: bool = False) -> ast.AST:
        """Original replacement-based mutations"""
        eligible_nodes = []
        
        for node in ast.walk(root):
            if not is_meaningful_node(node):
                continue
                
            if self.config.strategy == MutationStrategy.COMPREHENSION_FOCUSED:
                if isinstance(node, (ast.ListComp, ast.For, ast.comprehension)):
                    eligible_nodes.append(node)
            elif self.config.strategy == MutationStrategy.LAMBDA_FOCUSED:
                if isinstance(node, (ast.Lambda, ast.FunctionDef)):
                    eligible_nodes.append(node)
            elif self.config.strategy == MutationStrategy.CODE_AWARE:
                eligible_nodes.append(node)
            else:
                if map_ast_to_grammar_start(node):
                    eligible_nodes.append(node)
        
        if not eligible_nodes:
            print("No eligible nodes found for mutation")
            return root
        
        num_targets = 1 if single else min(self.config.num_mutations, len(eligible_nodes))
        target_nodes = random.sample(eligible_nodes, num_targets)
        
        replacer = NodeReplacer(self.config, {id(n) for n in target_nodes}, 
                               subtree_collector=self._subtree_collector)
        return replacer.visit(root)


class NodeReplacer(ast.NodeTransformer):
    """Handles node replacement during tree traversal"""
    
    def __init__(self, config: MutationConfig, target_ids: Set[int], 
                 replacement: Optional[ast.AST] = None,
                 subtree_collector: Optional[SubtreeCollector] = None):
        self.config = config
        self.target_ids = target_ids
        self.specific_replacement = replacement
        self.subtree_collector = subtree_collector
        self.mutations_done = 0

    def visit(self, node: ast.AST) -> ast.AST:
        if id(node) in self.target_ids:
            if self.specific_replacement:
                new_node = copy.deepcopy(self.specific_replacement)
            else:
                new_node = self.generate_replacement(node)
            
            print(f"\n--- Replacing {type(node).__name__} with {type(new_node).__name__} ---")
            if hasattr(node, 'name'):
                print(f"    Original: {node.name}")
            if hasattr(new_node, 'name'):
                print(f"    New: {new_node.name}")
            
            self.mutations_done += 1
            
            # Handle special cases
            if isinstance(node, ast.FunctionDef):
                if isinstance(new_node, ast.Lambda):
                    # Convert lambda to function, preserving original signature if requested
                    new_func = ast.FunctionDef(
                        name=node.name,
                        args=node.args if self.config.preserve_signatures else new_node.args,
                        body=[ast.Return(new_node.body)],
                        decorator_list=node.decorator_list if self.config.preserve_signatures else [],
                        returns=None
                    )
                    return ast.copy_location(new_func, node)
                elif isinstance(new_node, ast.FunctionDef) and self.config.preserve_signatures:
                    # Preserve original function signature
                    new_node.name = node.name
                    new_node.args = node.args
                    new_node.decorator_list = node.decorator_list
            
            if isinstance(node, ast.Expr) and not isinstance(new_node, ast.Expr):
                new_node = ast.Expr(value=new_node)
            
            return ast.copy_location(new_node, node)
        
        return self.generic_visit(node)

    def generate_replacement(self, node: ast.AST) -> ast.AST:
        """Generate replacement based on strategy"""
        try:
            # For code-aware strategy, try to reuse existing code
            if self.config.strategy == MutationStrategy.CODE_AWARE and self.subtree_collector:
                compatible = self.subtree_collector.get_compatible_subtrees(node)
                if compatible:
                    return copy.deepcopy(random.choice(compatible))
            
            # Strategy-specific generation
            if self.config.strategy == MutationStrategy.COMPREHENSION_FOCUSED:
                return find(simple_listcomp(), lambda x: True)
            elif self.config.strategy == MutationStrategy.LAMBDA_FOCUSED:
                if isinstance(node, ast.FunctionDef):
                    # Generate lambda that preserves function structure
                    return find(simple_lambda(preserve_args=node.args if self.config.preserve_signatures else None), 
                              lambda x: True)
                return find(simple_lambda(), lambda x: True)
            elif self.config.strategy == MutationStrategy.SIMPLE_IDENTIFIERS:
                return find(simple_statement(), lambda x: True)
            else:
                return self.generate_from_grammar(node)
        except Exception as e:
            print(f"Error generating replacement: {e}")
            return ast.Pass()

    def generate_from_grammar(self, node: ast.AST) -> ast.AST:
        """Generate replacement using grammar"""
        start_symbol = map_ast_to_grammar_start(node)
        if not start_symbol:
            return ast.Pass()
        
        try:
            grammar = Lark(
                LARK_GRAMMAR, 
                parser="lalr", 
                postlex=PythonIndenter(), 
                start="stmt" if start_symbol == "stmt" else start_symbol
            )
            strategy = EnhancedGrammarStrategy(grammar, start_symbol, self.config)
            
            new_code_str = find(strategy, lambda code: self.is_valid_code(code))
            new_ast = ast.parse(new_code_str)
            
            if new_ast.body:
                return new_ast.body[0]
            else:
                return ast.Pass()
        except Exception as e:
            print(f"Grammar generation failed: {e}")
            return ast.Pass()

    def is_valid_code(self, code: str) -> bool:
        """Check if code is valid Python"""
        try:
            ast.parse(code)
            return True
        except:
            return False


# --- Helper Functions ---

def map_ast_to_grammar_start(node: ast.AST) -> Optional[str]:
    """Maps an AST node type to a corresponding Lark grammar start symbol."""
    if isinstance(node, ast.stmt):
        return "stmt"
    if isinstance(node, ast.expr):
        return "eval_input"
    return None


def read_source_code(filepath: str) -> str:
    """Reads a Python source file and returns its content."""
    try:
        with open(filepath, "r", encoding="utf8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def generate_mutated_code(source_code: str, config: MutationConfig) -> str:
    """Generate mutated code based on configuration"""
    if sys.version_info < (3, 9):
        raise RuntimeError("This function requires Python 3.9+ for 'ast.unparse'.")
    
    try:
        original_ast = ast.parse(source_code)
    except SyntaxError as e:
        print(f"Error: The provided source code has a syntax error. {e}")
        return source_code

    mutator = EnhancedCodeMutator(config)
    mutated_ast = mutator.perform_mutations(original_ast)
    
    fix_missing_locations(mutated_ast)
    
    try:
        return ast.unparse(mutated_ast)
    except Exception as e:
        print(f"Error unparsing mutated AST: {e}")
        return source_code


# --- Example Usage ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mutator.py <path_to_python_file> [options]")
        print("Options:")
        print("  --strategy [grammar|comprehension|lambda|simple|subtree|codeaware]")
        print("  --operation [replace|add|delete|swap|reuse] (for subtree strategy)")
        print("  --mutations <number>")
        print("  --seed <number>")
        print("  --filter-keywords keyword1,keyword2,...")
        print("  --no-preserve-signatures  (don't preserve function signatures)")
        sys.exit(1)
    
    source_filepath = sys.argv[1]
    
    # Parse command line options
    config = MutationConfig()
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--strategy" and i + 1 < len(sys.argv):
            config.strategy = MutationStrategy(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--operation" and i + 1 < len(sys.argv):
            config.operation = MutationOperation(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--mutations" and i + 1 < len(sys.argv):
            config.num_mutations = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--seed" and i + 1 < len(sys.argv):
            config.seed = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--filter-keywords" and i + 1 < len(sys.argv):
            config.filtered_keywords = set(sys.argv[i + 1].split(','))
            i += 2
        elif sys.argv[i] == "--no-preserve-signatures":
            config.preserve_signatures = False
            i += 1
        else:
            print(f"Unknown option: {sys.argv[i]}")
            i += 1
    
    print(f"🧬 Mutation Configuration:")
    print(f"  Strategy: {config.strategy.value}")
    if config.strategy == MutationStrategy.SUBTREE_MANIPULATION:
        print(f"  Operation: {config.operation.value}")
    print(f"  Number of mutations: {config.num_mutations}")
    print(f"  Seed: {config.seed}")
    print(f"  Filtered keywords: {config.filtered_keywords}")
    print(f"  Preserve signatures: {config.preserve_signatures}")
    
    print(f"\n📖 Reading original code from: {source_filepath}")
    original_code = read_source_code(source_filepath)
    
    print("\nOriginal Code:")
    print("=" * 40)
    print(original_code)
    print("=" * 40)
    
    mutated_code = generate_mutated_code(original_code, config)
    
    print("\n✅ Generated Mutated Code:")
    print("=" * 40)
    print(mutated_code)
    print("=" * 40)