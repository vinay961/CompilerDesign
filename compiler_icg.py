class ICGenerator:
    def __init__(self):
        self.ic_code = []

    def generate(self, ast):
        for statement in ast:
            self.generate_statement(statement)

    def generate_statement(self, statement):
        if statement[0] == 'ASSIGN':
            var_name = statement[1]
            expr = statement[2]
            code = f"{var_name} = {self.generate_expression(expr)}"
            self.ic_code.append(code)

        elif statement[0] == 'FOR':
            var_name = statement[1]
            start_expr = statement[2]
            condition = statement[3]
            increment = statement[4]
            body = statement[5]
            self.ic_code.append(f"for {var_name} = {self.generate_expression(start_expr)} to {self.generate_expression(condition)} do:")
            self.generate(body)

    def generate_expression(self, expr):
        if isinstance(expr, tuple):
            left, op, right = expr
            return f"({self.generate_expression(left)} {op} {self.generate_expression(right)})"
        else:
            return str(expr)
