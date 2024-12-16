class Executor:
    def __init__(self):
        self.variables = {}

    def execute(self, ic_code):
        for code in ic_code:
            exec(code, {}, self.variables)
        print("Execution result:", self.variables)
