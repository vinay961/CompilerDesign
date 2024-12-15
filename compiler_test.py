import re

# --- Phase 1: Lexical Analysis (Tokenization) ---
def lexical_analysis(source_code):
    """
    Breaks source code into tokens.
    Uses a regular expression to find keywords, identifiers, numbers,
    operators, and symbols.
    """
    tokens = re.findall(
        r'\bint\b|\bif\b|\belse\b|\bfor\b|\bwhile\b|\bprint\b|'
        r'[0-9]+(?:\.[0-9]*)?|\b[A-Za-z_][A-Za-z0-9_]*\b|'
        r'==|!=|>=|<=|>|<|\+|[-/*=;(){},]', source_code)
    print("Tokens:", tokens)
    return tokens

# --- Phase 2: Syntax Analysis (Parsing) ---
class ASTNode:
    """
    Represents a node in the Abstract Syntax Tree (AST).
    Each node has a type, an optional value, and a list of children.
    """
    def __init__(self, type, value=None, children=None):
        self.type = type
        self.value = value
        self.children = children if children else []

    def __repr__(self, level=0):
        """
        Provides a string representation of the AST node, making it
        easy to visualize the tree structure.
        """
        indent = '  ' * level
        node_str = f"{indent}{self.type}"
        if self.value is not None:
            node_str += f"({self.value})"
        if self.children:
            for child in self.children:
                node_str += "\n" + child.__repr__(level + 1)
        return node_str

class Parser:
    """
    Parses the tokens and creates an Abstract Syntax Tree (AST).
    The AST represents the structure of the code.
    """
    def __init__(self, tokens):
        self.tokens = tokens
        self.current_token_index = 0

    def _peek(self):
      """Returns the current token without consuming it."""
      if self.current_token_index < len(self.tokens):
          return self.tokens[self.current_token_index]
      return None

    def _consume(self):
        """Returns the current token and advances the index to the next token."""
        if self.current_token_index < len(self.tokens):
            token = self.tokens[self.current_token_index]
            self.current_token_index += 1
            return token
        return None

    def _match(self, expected_token):
        """Checks if the current token matches the expected token.
        If they match, consumes the current token and returns True,
        otherwise returns False.
        """
        actual_token = self._peek()
        if actual_token == expected_token:
            self._consume()
            return True
        return False

    def _expect(self, expected_token):
       """
       Checks if the current token matches the expected token.
       If they match, it consumes the current token.
       If not, it raises a ValueError.
       """
       if not self._match(expected_token):
            actual_token = self._peek()
            raise ValueError(f"Expected '{expected_token}', but got '{actual_token}'")


    def parse(self):
        """Starts the parsing process."""
        return self.parse_program()

    def parse_program(self):
        """Parses the entire program. Creates a PROGRAM node that contains
        a list of all statements.
        """
        program_node = ASTNode("PROGRAM")
        while self._peek() is not None:
            program_node.children.append(self.parse_statement())
        print("\nAbstract Syntax Tree:")
        print(program_node)
        return program_node

    def parse_statement(self):
      """
      Parses a single statement. A statement can be a declaration,
      an if statement, a while loop, a for loop, a print statement,
      or an expression statement.
      """
      if self._peek() in ('int'):
          return self.parse_declaration()
      elif self._peek() == 'if':
          return self.parse_if_statement()
      elif self._peek() == 'while':
           return self.parse_while_statement()
      elif self._peek() == 'for':
           return self.parse_for_statement()
      elif self._peek() == 'print':
          return self.parse_print_statement()
      else:
         return self.parse_expression_statement()

    def parse_expression_statement(self):
        """Parses an expression statement, which ends with a semicolon."""
        node = self.parse_expression()
        self._expect(";")
        return node

    def parse_declaration(self):
        """Parses a variable declaration statement (e.g., 'int x;' or 'int x = 10;')."""
        data_type = self._consume()
        variable_name = self._consume()
        if self._match("="):
            expression_node = self.parse_expression()
            self._expect(";")
            return ASTNode("DECLARATION_ASSIGNMENT", value=(data_type, variable_name), children=[expression_node])
        else:
            self._expect(";")
            return ASTNode("DECLARATION", value=(data_type, variable_name))

    def parse_if_statement(self):
       """Parses an if statement."""
       self._expect("if")
       self._expect("(")
       condition_node = self.parse_expression()
       self._expect(")")
       self._expect("{")
       then_block = self.parse_block()
       else_block = None
       if self._match("else"):
         self._expect("{")
         else_block = self.parse_block()
       if else_block is None:
           return ASTNode("IF", children=[condition_node, then_block])
       else:
           return ASTNode("IF", children=[condition_node, then_block, else_block])

    def parse_while_statement(self):
       """Parses a while loop statement."""
       self._expect("while")
       self._expect("(")
       condition_node = self.parse_expression()
       self._expect(")")
       self._expect("{")
       while_block = self.parse_block()
       return ASTNode("WHILE", children=[condition_node, while_block])

    def parse_for_statement(self):
      """Parses a for loop statement."""
      self._expect("for")
      self._expect("(")
      
      # Handle initialization statement inside for loop
      init_node = None
      if self._peek() in ("int"):
          init_node = self.parse_declaration()
      else:
          init_node = self.parse_expression()
      self._expect(";")
      
      condition_node = self.parse_expression()
      self._expect(";")
      increment_node = self.parse_expression()
      self._expect(")")
      self._expect("{")
      for_block = self.parse_block()
      if init_node:
        return ASTNode("FOR", children=[init_node, condition_node, increment_node, for_block])
      else:
        return ASTNode("FOR", children=[ASTNode("EMPTY"), condition_node, increment_node, for_block])


    def parse_print_statement(self):
       """Parses a print statement."""
       self._expect("print")
       self._expect("(")
       expression_node = self.parse_expression()
       self._expect(")")
       self._expect(";")
       return ASTNode("PRINT", children=[expression_node])

    def parse_block(self):
      """Parses a block of code inside curly braces."""
      block_node = ASTNode("BLOCK")
      while self._peek() is not None and self._peek() != '}':
        block_node.children.append(self.parse_statement())
      self._expect("}")
      return block_node

    def parse_expression(self):
      """Parses an expression."""
      return self.parse_assignment_expression()

    def parse_assignment_expression(self):
      """Parses an assignment expression."""
      left = self.parse_comparison_expression()
      if self._match("="):
          right = self.parse_assignment_expression()
          return ASTNode("ASSIGN", children=[left, right])
      return left
    
    def parse_comparison_expression(self):
        """Parses a comparison expression."""
        left = self.parse_additive_expression()
        while self._peek() in ('==', '!=', '>', '<', '>=', '<='):
            op = self._consume()
            right = self.parse_additive_expression()
            left = ASTNode(op, children=[left, right])
        return left

    def parse_additive_expression(self):
       """Parses an additive expression."""
       left = self.parse_term()
       while self._peek() in ('+', '-'):
           op = self._consume()
           right = self.parse_term()
           left = ASTNode(op, children=[left, right])
       return left

    def parse_term(self):
       """Parses a term."""
       left = self.parse_factor()
       while self._peek() in ('*', '/'):
           op = self._consume()
           right = self.parse_factor()
           left = ASTNode(op, children=[left, right])
       return left

    def parse_factor(self):
      """Parses a factor, which can be a number, identifier, or an expression in parentheses."""
      token = self._peek()
      if token is not None:
          if token.replace(".", "", 1).isdigit() or (token[0] == '-' and token[1:].replace(".", "", 1).isdigit()):
             self._consume()
             return ASTNode("NUMBER", value=token)
          elif token.isalnum() and token.isalpha():
            self._consume()
            return ASTNode("IDENTIFIER", value=token)
          elif self._match("("):
              expression = self.parse_expression()
              self._expect(")")
              return expression
      
      raise ValueError(f"Expected a factor, but got '{token}'")


# --- Phase 3: Semantic Analysis ---
class SemanticAnalyzer:
  """
  Analyzes the Abstract Syntax Tree (AST).
  Checks for semantic errors, such as undeclared variables.
  Creates a symbol table, which stores information about variables.
  """
  def __init__(self, ast):
    self.ast = ast
    self.symbol_table = {}

  def analyze(self):
    """Begins the semantic analysis of the AST."""
    self._analyze_node(self.ast)
    print("\nSymbol Table:", self.symbol_table)
    return self.symbol_table, self.ast
    
  def _analyze_node(self, node):
    """Recursively analyzes each node in the AST."""
    if node.type == "PROGRAM":
      for child in node.children:
        self._analyze_node(child)
    elif node.type == "DECLARATION":
       data_type, var_name = node.value
       if var_name in self.symbol_table:
           raise ValueError(f"Variable '{var_name}' already declared.")
       self.symbol_table[var_name] = {"type": data_type}
    elif node.type == "DECLARATION_ASSIGNMENT":
        data_type, var_name = node.value
        if var_name in self.symbol_table:
            raise ValueError(f"Variable '{var_name}' already declared.")
        self.symbol_table[var_name] = {"type": data_type}
        self._analyze_node(node.children[0])
    elif node.type == "IDENTIFIER":
        if node.value not in self.symbol_table:
           raise ValueError(f"Variable '{node.value}' not declared")
    elif node.type in ("IF","WHILE","FOR"):
      for child in node.children:
        self._analyze_node(child)
    elif node.type == "BLOCK":
      for child in node.children:
         self._analyze_node(child)
    elif node.type == "PRINT":
        self._analyze_node(node.children[0])
    elif node.children:
        for child in node.children:
            self._analyze_node(child)

# --- Phase 4: Intermediate Code Generation ---
class IntermediateCodeGenerator:
    """
    Translates the AST into intermediate code instructions.
    Uses a list to store the generated instructions.
    """
    def __init__(self, ast, symbol_table):
        self.ast = ast
        self.symbol_table = symbol_table
        self.intermediate_code = []
        self.label_count = 0
    
    def generate(self):
       """Starts the intermediate code generation process."""
       self._generate_node(self.ast)
       print("\nIntermediate Code:", self.intermediate_code)
       return self.intermediate_code

    def _generate_node(self, node):
        """Recursively generates intermediate code for each node in the AST."""
        if node.type == "PROGRAM":
           for child in node.children:
               self._generate_node(child)
        elif node.type == "DECLARATION":
            _, var_name = node.value
            self.intermediate_code.append(f"ALLOCATE {var_name}")
        elif node.type == "DECLARATION_ASSIGNMENT":
           _, var_name = node.value
           self.intermediate_code.append(f"ALLOCATE {var_name}")
           self._generate_node(node.children[0])
           self.intermediate_code.append(f"STORE {var_name}")
        elif node.type == "ASSIGN":
            self._generate_node(node.children[1])
            temp_var = f"temp_{self.label_count}"
            self.label_count += 1
            self.intermediate_code.append(f"STORE {temp_var}")
            self._generate_node(node.children[0])
            self.intermediate_code.append(f"LOAD {temp_var}")
            self.intermediate_code.append(f"STORE {node.children[0].value}")
        elif node.type == "NUMBER":
            self.intermediate_code.append(f"LOAD {node.value}")
        elif node.type == "IDENTIFIER":
            self.intermediate_code.append(f"LOAD {node.value}")
        elif node.type in ('+', '-', '*', '/', '==', '!=', '>', '<', '>=', '<='):
            self._generate_node(node.children[0])
            self.intermediate_code.append("PUSH")
            self._generate_node(node.children[1])
            self.intermediate_code.append("POP")
            self.intermediate_code.append(f"OPERATION {node.type}")
        elif node.type == "PRINT":
             self._generate_node(node.children[0])
             self.intermediate_code.append(f"PRINT")
        elif node.type == "IF":
            label_if_end = f"L{self.label_count}"
            self.label_count += 1
            self._generate_node(node.children[0])
            self.intermediate_code.append(f"IF_FALSE GOTO {label_if_end}")
            self._generate_node(node.children[1])
            if len(node.children) == 3:
              label_else_end = f"L{self.label_count}"
              self.label_count +=1
              self.intermediate_code.append(f"GOTO {label_else_end}")
              self.intermediate_code.append(f"{label_if_end}:")
              self._generate_node(node.children[2])
              self.intermediate_code.append(f"{label_else_end}:")
            else:
              self.intermediate_code.append(f"{label_if_end}:")
        elif node.type == "WHILE":
            label_while_start = f"L{self.label_count}"
            self.label_count +=1
            label_while_end = f"L{self.label_count}"
            self.label_count +=1
            self.intermediate_code.append(f"{label_while_start}:")
            self._generate_node(node.children[0])
            self.intermediate_code.append(f"IF_FALSE GOTO {label_while_end}")
            self._generate_node(node.children[1])
            self.intermediate_code.append(f"GOTO {label_while_start}")
            self.intermediate_code.append(f"{label_while_end}:")
        elif node.type == "FOR":
             label_for_start = f"L{self.label_count}"
             self.label_count += 1
             label_for_end = f"L{self.label_count}"
             self.label_count += 1

             self._generate_node(node.children[0]) # init
             self.intermediate_code.append(f"{label_for_start}:")
             self._generate_node(node.children[1]) #condition
             self.intermediate_code.append(f"IF_FALSE GOTO {label_for_end}")
             self._generate_node(node.children[3]) #body
             self._generate_node(node.children[2]) #increment
             self.intermediate_code.append(f"GOTO {label_for_start}")
             self.intermediate_code.append(f"{label_for_end}:")
        elif node.type == "BLOCK":
            for child in node.children:
                self._generate_node(child)
        elif node.type == "EMPTY":
            pass


# --- Phase 5: Code Optimization ---
def code_optimization(intermediate_code):
    """
    Placeholder for code optimization.
    Currently, it simply returns the intermediate code without any changes.
    """
    optimized_code = intermediate_code
    print("\nOptimized Code:", optimized_code)
    return optimized_code

# --- Phase 6: Code Generation ---
class CodeGenerator:
    """
    Generates the final code in a simplified assembly-like language.
    Uses registers (R0 and R1).
    """
    def __init__(self, optimized_code):
        self.optimized_code = optimized_code
        self.registers = {"R0": None, "R1":None}
        self.memory = {}
        self.next_register = "R0"

    def generate(self):
        """Generates the final code."""
        final_code = []
        for line in self.optimized_code:
           parts = line.split()
           if parts[0] == "LOAD":
              if parts[1] in self.memory:
                  final_code.append(f"mov {parts[1]}, {self.next_register}")
              else:
                 final_code.append(f"mov_immediate {parts[1]} {self.next_register}")
           elif parts[0] == "STORE":
                final_code.append(f"mov {self.next_register}, {parts[1]}")
                if self.next_register == "R0":
                  self.next_register = "R1"
                else:
                  self.next_register = "R0"
           elif parts[0] == "ALLOCATE":
              self.memory[parts[1]] = 0
              final_code.append(f"allocate {parts[1]}")
           elif parts[0] == "PUSH":
                final_code.append(f"push {self.next_register}")
                if self.next_register == "R0":
                  self.next_register = "R1"
                else:
                  self.next_register = "R0"
           elif parts[0] == "POP":
               if self.next_register == "R0":
                  self.next_register = "R1"
               else:
                  self.next_register = "R0"
               final_code.append(f"pop {self.next_register}")
           elif parts[0] == "OPERATION":
             op = parts[1]
             if op == "+":
                final_code.append(f"add R0, R1")
             elif op == "-":
                final_code.append(f"sub R0, R1")
             elif op == "*":
                final_code.append(f"mul R0, R1")
             elif op == "/":
                final_code.append(f"div R0, R1")
             elif op == "==":
                 final_code.append(f"eq R0, R1")
             elif op == "!=":
                final_code.append(f"neq R0, R1")
             elif op == "<":
               final_code.append(f"lt R0, R1")
             elif op == ">":
                final_code.append(f"gt R0, R1")
             elif op == "<=":
                final_code.append(f"lte R0, R1")
             elif op == ">=":
                 final_code.append(f"gte R0, R1")

           elif parts[0] == "PRINT":
              final_code.append(f"print R0")
           elif parts[0] == "IF_FALSE":
              final_code.append(f"if_false R0 jump {parts[2]}")
           elif parts[0] == "GOTO":
               final_code.append(f"goto {parts[1]}")
           elif parts[0].startswith("L"):
               final_code.append(line)
        print("\nFinal Code:", final_code)
        return final_code

# --- Function to execute the final code ---
def execute_final_code(final_code):
    """
    Executes the final code (in the simplified assembly-like language).
    Simulates the operations on memory and registers.
    """
    memory = {}
    registers = {"R0": 0, "R1": 0}
    output = []
    ip = 0

    def get_value(operand):
      """
      Retrieves the value of an operand, which can be a memory location,
      a register, or a literal number.
      """
      if operand in memory:
          return memory[operand]
      elif operand in registers:
          return registers[operand]
      else:
          return float(operand)

    while ip < len(final_code):
       instruction = final_code[ip]
       parts = instruction.split()
       if parts[0] == "allocate":
           memory[parts[1]] = 0
       elif parts[0] == "mov_immediate":
            registers[parts[2]] = float(parts[1])
       elif parts[0] == "mov":
           if parts[1].endswith(','): # fix for case R0, and R1,
              registers[parts[2]] = get_value(parts[1][:-1])
           else:
              registers[parts[2]] = get_value(parts[1])
       elif parts[0] == "add":
            registers["R0"] = get_value("R0") + get_value("R1")
       elif parts[0] == "sub":
            registers["R0"] = get_value("R0") - get_value("R1")
       elif parts[0] == "mul":
           registers["R0"] = get_value("R0") * get_value("R1")
       elif parts[0] == "div":
            if get_value("R1") != 0:
              registers["R0"] = get_value("R0") / get_value("R1")
       elif parts[0] == "eq":
          registers["R0"] = 1 if get_value("R0") == get_value("R1") else 0
       elif parts[0] == "neq":
          registers["R0"] = 1 if get_value("R0") != get_value("R1") else 0
       elif parts[0] == "lt":
            registers["R0"] = 1 if get_value("R0") < get_value("R1") else 0
       elif parts[0] == "gt":
           registers["R0"] = 1 if get_value("R0") > get_value("R1") else 0
       elif parts[0] == "lte":
           registers["R0"] = 1 if get_value("R0") <= get_value("R1") else 0
       elif parts[0] == "gte":
           registers["R0"] = 1 if get_value("R0") >= get_value("R1") else 0
       elif parts[0] == "push":
         if parts[1] == "R0":
           registers["R1"] = registers["R0"]
         elif parts[1] == "R1":
            registers["R0"] = registers["R1"]
       elif parts[0] == "pop":
         if parts[1] == "R0":
           registers["R0"] = registers["R1"]
         elif parts[1] == "R1":
            registers["R1"] = registers["R0"]
       elif parts[0] == "print":
            output.append(registers["R0"])
       elif parts[0] == "if_false":
            if registers["R0"] == 0:
              ip = final_code.index(parts[3]+":")
              continue
       elif parts[0] == "goto":
            ip = final_code.index(parts[1]+":")
            continue
       elif parts[0].startswith("L"):
            pass

       ip += 1
    print("\nExecuted Output:", output)
    return output

# --- Example usage ---
source_code = """
int a = 5;
int b = 10;
int c = a + b;
if (c > 10) {
    print(c);
} else {
    print(a);
}
"""

source_code1 = """
int x = 10;
int y = 20;
int z = x + y;
print (z);
"""


# Phase 1: Lexical Analysis
tokens = lexical_analysis(source_code)

# Phase 2: Syntax Analysis
parser = Parser(tokens)
ast = parser.parse()

# Phase 3: Semantic Analysis
semantic_analyzer = SemanticAnalyzer(ast)
symbol_table, ast = semantic_analyzer.analyze()

# Phase 4: Intermediate Code Generation
intermediate_code_generator = IntermediateCodeGenerator(ast, symbol_table)
intermediate_code = intermediate_code_generator.generate()

# Phase 5: Code Optimization
optimized_code = code_optimization(intermediate_code)

# Phase 6: Code Generation
code_generator = CodeGenerator(optimized_code)
final_code = code_generator.generate()


# Phase 7: Execute the final code
# execute_final_code(final_code)
