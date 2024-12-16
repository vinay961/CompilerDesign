def log_to_file(lexer_tokens, parsed_code, icg_instructions, execution_result):
    with open("compiler_log.txt", "w") as f:
        f.write("Lexer Tokens:\n")
        for token in lexer_tokens:
            f.write(f'{token}\n')
        
        f.write("\nParsed Code:\n")
        f.write(parsed_code + '\n')
        
        f.write("\nIntermediate Code:\n")
        for instruction in icg_instructions:
            f.write(f'{instruction}\n')

        f.write("\nExecution Result:\n")
        for var, value in execution_result.items():
            f.write(f'{var} = {value}\n')
