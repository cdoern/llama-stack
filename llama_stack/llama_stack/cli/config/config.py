import argparse
from llama_stack.cli.subcommand import Subcommand

class ConfigParser(Subcommand):
    """Llama cli for configuration interface apis"""


    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "config",
            prog=
        )