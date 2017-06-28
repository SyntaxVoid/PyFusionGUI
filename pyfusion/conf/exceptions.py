"""Exceptions for the pyfusion config parser."""

class DisallowedSectionType(Exception):
    """Config file includes a section type which is not allowed."""
    def __init__(self, section_name):
        self.section_name = section_name
    def __str__(self):
        return repr(self.section_name)

class ConfigSectionSyntaxError(Exception):
    """Config file has a malformed section label."""
    def __init__(self, section_name):
        self.section_name = section_name
    def __str__(self):
        return repr(self.section_name)


class UnknownVariableTypeError(Exception):
    """Cannot recognise requested variable type in conifg."""
    pass
