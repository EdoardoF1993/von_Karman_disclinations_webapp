import sys

class Tee:
    def __init__(self, stdout_filename, stderr_filename):
        self.stdout_file = open(stdout_filename, "w")  # Open file for stdout
        self.stderr_file = open(stderr_filename, "w")  # Open file for stderr
        self.terminal_stdout = sys.stdout  # Store original stdout
        self.terminal_stderr = sys.stderr  # Store original stderr

    def write_stdout(self, message):
        self.terminal_stdout.write(message)  # Print to terminal
        self.stdout_file.write(message)  # Save to stdout.txt

    def write_stderr(self, message):
        self.terminal_stderr.write(message)  # Print to terminal
        self.stderr_file.write(message)  # Save to stderr.txt

    def flush(self):
        self.terminal_stdout.flush()
        self.stdout_file.flush()
        self.terminal_stderr.flush()
        self.stderr_file.flush()
