import pyfusion as pf

pf.config.get('global','database')  #  'sqlite:///sqlite.txt'
pf.read_config('shaun_feb_2010.cfg')

from pyfusion.conf.utils import get_config_as_dict
get_config_as_dict('Device','H1')

get_config_as_dict('Diagnostic','H1PoloidalAll')
