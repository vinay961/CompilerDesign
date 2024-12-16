class SemanticAnalyzer:
    def __init__(self):
        self.symbol_table = {}

    def analyze(self, ast):
        for statement in ast:
            self.check_statement(statement)

    def check_statement(self, statement):
        if statement[0] == 'ASSIGN':
            var_name = statement[1]
            expr = statement[2]
            if var_name not in self.symbol_table:
                raise NameError(f"Variable '{var_name}' is not defined")
            self.check_expression(expr)

        elif statement[0] == 'FOR':
            var_name = statement[1]
            if var_name in self.symbol_table:
                raise NameError(f"Variable '{var_name}' is already defined")
            self.symbol_table[var_name] = 'int'
            start_expr = statement[2]
            condition = statement[3]
            increment = statement[4]
            self.check_expression(start_expr)
            self.check_expression(condition)
            self.check_expression(increment)
            self.analyze(statement[5])  # Analyze loop body

    def check_expression(self, expr):
        if isinstance(expr, tuple):  # Binary expression (e.g., x + 2)
            left, operator, right = expr
            self.check_expression(left)
            self.check_expression(right)
        elif isinstance(expr, str):  # Variable reference
            if expr not in self.symbol_table:
                raise NameError(f"Variable '{expr}' is not defined")
