import ply.lex as lex

# List of token names
tokens = (
    'INT', 'IF', 'ELSE', 'FOR', 'WHILE', 'PRINT',  # Keywords
    'ID', 'NUMBER',                               # Identifiers and Numbers
    'EQ', 'NE', 'GE', 'LE', 'GT', 'LT',           # Comparison Operators
    'PLUS', 'MINUS', 'MULT', 'DIV', 'ASSIGN',     # Arithmetic Operators
    'SEMI', 'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'COMMA'  # Symbols
)

# Regular expression rules for simple tokens
t_EQ = r'=='
t_NE = r'!='
t_GE = r'>='
t_LE = r'<='
t_GT = r'>'
t_LT = r'<'
t_PLUS = r'\+'
t_MINUS = r'-'
t_MULT = r'\*'
t_DIV = r'/'
t_ASSIGN = r'='
t_SEMI = r';'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACE = r'\{'
t_RBRACE = r'\}'
t_COMMA = r','

# Rules with actions
def t_INT(t):
    r'\bint\b'
    return t

def t_IF(t):
    r'\bif\b'
    return t

def t_ELSE(t):
    r'\belse\b'
    return t

def t_FOR(t):
    r'\bfor\b'
    return t

def t_WHILE(t):
    r'\bwhile\b'
    return t

def t_PRINT(t):
    r'\bprint\b'
    return t

def t_NUMBER(t):
    r'[0-9]+(?:\.[0-9]*)?'
    t.value = float(t.value) if '.' in t.value else int(t.value)  # Convert to int or float
    return t

def t_ID(t):
    r'\b[A-Za-z_][A-Za-z0-9_]*\b'
    return t

# Ignored characters (e.g., whitespace)
t_ignore = ' \t'

# Error handling rule
def t_error(t):
    print(f"Illegal character '{t.value[0]}' at line {t.lineno}")
    t.lexer.skip(1)

# Build the lexer
lexer = lex.lex()

def lexical_analysis(source_code):
    lexer.input(source_code)
    tokens = []
    while True:
        tok = lexer.token()
        if not tok:
            break  # No more input
        tokens.append(tok)
    return tokens
