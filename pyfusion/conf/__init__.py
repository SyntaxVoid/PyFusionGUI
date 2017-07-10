"""Tools for processing configuration files."""

from ConfigParser import ConfigParser

from pyfusion.conf.exceptions import DisallowedSectionType, \
     ConfigSectionSyntaxError, UnknownVariableTypeError

# This list contains allowed section types, i.e. [SectionType:Name] in
# config files. Be sure to update the documentation (config.rst) when
# adding to this list
allowed_section_types = ['Device', 'Diagnostic', 'Acquisition', 'CoordTransform']
## sections which don't follow the [SectionType:Name] syntax
special_section_names = ['variabletypes', 'global', 'Plots']


class PyfusionConfigParser(ConfigParser):
    """Customised parser to facilitate [Type:Name] config sections.
    
    Inherited ConfigParser methods are extended, and prefixed with pf_
    to allow separate arguments for section type and section name,
    for example:

      ConfigParser.has_section(sectionname) ->
      PyfusionConfigParser.pf_has_section(sectiontype, name)

    The inherited ConfigParser methods are still available, so the
    following are identical:

      PyfusionConfigParser.has_section('Device:TJII')
      PyfusionConfigParser.pf_has_section('Device','TJII')
          
    """

    #####################################################
    ## Extensions to ConfigParser methods (incomplete) ##
    #####################################################
    
    def pf_has_section(self, sectiontype, sectionname):
        return self.has_section("%s:%s"%(sectiontype, sectionname))

    def pf_get(self, sectiontype, sectionname, option):
        if self.has_option('variabletypes',
                           "%s__%s"%(sectiontype, option)):
            output_type = self.get('variabletypes',
                                   "%s__%s"%(sectiontype, option))
            if output_type == 'float':
                return self.getfloat("%s:%s"%(sectiontype, sectionname), option)
            elif output_type == 'int':
                return self.getint("%s:%s"%(sectiontype, sectionname), option)
            elif output_type == 'bool':
                return self.getboolean("%s:%s"%(sectiontype, sectionname), option)
            else:
                raise UnknownVariableTypeError
        else:
            return self.get("%s:%s"%(sectiontype, sectionname), option)

    def pf_options(self, sectiontype, sectionname):
        return self.options("%s:%s"%(sectiontype, sectionname))

    def pf_has_option(self, sectiontype, sectionname, option):
        return self.has_option("%s:%s"%(sectiontype, sectionname), option)

    #########################################
    ## Custom PyfusionConfigParser methods ##
    #########################################

    def check_section_syntax(self):
        """Make sure config file sections follow [Type:Name] syntax."""
        for section in self.sections():
            if not section in special_section_names:
                split_name = section.split(':')
                if not len(split_name)==2:
                    raise ConfigSectionSyntaxError, section
                if not (len(split_name[0])>0 and len(split_name[1])>0):
                    raise ConfigSectionSyntaxError, section

    def check_section_types(self, type_list):
        """Make sure section types listed in config file are allowed."""
        self.check_section_syntax()
        for section in self.sections():
            if not section in special_section_names:
                section_name = section.split(':')[0]
                if not section_name in type_list:
                    raise DisallowedSectionType, section_name

#config = PyfusionConfigParser()

