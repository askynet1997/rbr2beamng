import configparser


def config_parser() -> configparser.ConfigParser:
    parser = configparser.ConfigParser(
        interpolation=None,
        strict=False,
        delimiters=("=",),
        comment_prefixes=(";", "#"),
        empty_lines_in_values=False,
    )
    parser.optionxform = str
    return parser
