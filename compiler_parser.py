class Parser:
    def __init__(self, tokens):
        self.tokens = tokens  # List of tokens
        self.current = 0      # Current position in the token list

    def current_token(self):
        """Returns the current token."""
        return self.tokens[self.current]

    def match(self, token_type):
        """Matches and consumes the current token if it matches the given token type."""
        token = self.current_token()
        if token[0] == token_type:
            self.current += 1
        else:
            raise SyntaxError(f"Expected {token_type}, but found {token[1]} at position {self.current}")

    def peek_token(self):
        """Peeks at the next token without consuming it."""
        if self.current + 1 < len(self.tokens):
            return self.tokens[self.current + 1]
        return None

    def statement(self):
        """Handles different statements (variable declarations, assignments, etc.)."""
        token = self.current_token()

        # Handle assignment (e.g., result = x + y;)
        if token[0] == 'ID':  # variable name (e.g., result)
            self.match('ID')  # Consume variable name (e.g., result)
            if self.peek_token() and self.peek_token()[0] == 'ASSIGN':  # Check for '=' operator
                self.match('ASSIGN')  # Consume '='
                self.expression()  # Handle the right-hand side of the assignment (e.g., x + y)
            if self.peek_token() and self.peek_token()[0] == 'SEMI':  # Check for semicolon at the end
                self.match('SEMI')  # Consume ';'
            return 'simple assignment'

        # Raise error if no known statement type matches
        else:
            raise SyntaxError(f"Unexpected token {token[1]} at position {self.current}")

    def expression(self):
        """Handles expressions (operations like +, -, *, /, etc.)."""
        self.term()  # Start parsing with terms (e.g., x, y)

        while self.peek_token() and self.peek_token()[0] == 'OP':  # Check for operator (+, -, etc.)
            self.match('OP')  # Consume operator
            self.term()  # Parse the next term

    def term(self):
        """Handles terms in the expression (numbers, variables, etc.)."""
        token = self.current_token()
        if token[0] == 'ID':  # Check for variable (e.g., x, y)
            self.match('ID')  # Consume variable
        elif token[0] == 'NUMBER':  # Check for a number
            self.match('NUMBER')  # Consume number
        else:
            raise SyntaxError(f"Unexpected token {token[1]} at position {self.current}")

    def parse(self):
        """Starts the parsing process by handling the first statement."""
        statements = []
        while self.current < len(self.tokens):
            statements.append(self.statement())  # Parse individual statements
        return statements
