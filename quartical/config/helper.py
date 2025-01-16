import sys
import os
import textwrap
import re
from colorama import Fore, Style
from omegaconf import OmegaConf as oc
from dataclasses import fields
from quartical.config.external import finalize_structure

GAIN_MSG = (
    "Gains make use of a special configuration mechanism. Use 'solver.terms' "
    "to specify the name of each gain term, e.g. 'solver.terms=[G,B]'. Each "
    "gain can then be configured using its name and any gain option, e.g. "
    "'G.type=complex' or 'B.direction_dependent=True'."
)

HELP_MSG = (
    "For full help, use 'goquartical help'. For help with a specific "
    "section, use e.g. 'goquartical help='[section1, section2]''. Help is "
    "available for {}. Other command line utilitites: [goquartical-backup, "
    "goquartical-restore]."
)


def print_help(HelpConfig, section_names=None):

    # This guards against attempting to get the terminal size when the output
    # is being piped/redirected.
    if sys.stdout.isatty():
        columns, _ = os.get_terminal_size(0)
    else:
        columns = 80  # Fall over to some sensible default.

    help_message = f"{Style.BRIGHT}"

    all_section_names = [fld.name for fld in fields(HelpConfig)]
    section_names = section_names or all_section_names

    for section_name in section_names:

        section = getattr(HelpConfig, section_name)
        help_message += f"\n{Fore.CYAN}{section_name:-^{columns}}\n"

        if section_name == "gain":
            txt = textwrap.fill(GAIN_MSG, width=columns)
            help_message += f"{Fore.GREEN}{txt:-^{columns}}\n"

        for key, value in section.__helpstr__().items():
            option = f"{section_name}.{key}"
            help_message += f"{Fore.MAGENTA}{option:<}\n"
            txt = textwrap.fill(
                value,
                width=columns,
                initial_indent=" "*4,
                subsequent_indent=" "*4
            )
            help_message += f"{Fore.WHITE}{txt:<{columns}}\n"

        help_message += f"{Fore.CYAN}{'':-^{columns}}\n"

    txt = textwrap.fill(HELP_MSG.format(all_section_names), width=columns)

    help_message += f"{Fore.GREEN}{txt:-^{columns}}\n"

    help_message += f"{Fore.CYAN}{'':-^{columns}}{Style.RESET_ALL}"

    print(help_message)


def help():
    """Prints the help."""

    help_args = [arg for arg in sys.argv if arg.startswith('help')]

    # Always take the last specified help request.
    help_arg = help_args.pop() if help_args else help_args

    # Early return when help is not required.
    if len(sys.argv) != 1 and not help_arg:
        return

    # Include a generic gain term in the help config.
    additional_config = [oc.from_dotlist(["solver.terms=['gain']"])]
    help_class = finalize_structure(additional_config)
    HelpConfig = help_class()

    if len(sys.argv) == 1 or help_arg == "help":
        print_help(HelpConfig)
    else:
        selection = help_arg.split("=")[-1]
        selection = re.sub(r'[\[\] ]', "", selection)
        selection = selection.split(",")
        print_help(HelpConfig, section_names=selection)

    sys.exit()
