from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution("SONATA").version
except DistributionNotFound:
    pass

from . import scotv1
from . import sonata
from . import util, vis, pipline

