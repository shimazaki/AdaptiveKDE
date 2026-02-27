from .sshist import sshist
from .sskernel import sskernel
from .ssvkernel import ssvkernel
from .classic import sshist_classic, sskernel_classic, ssvkernel_classic

__version__ = '1.2.0'

__all__ = ('sshist', 'sskernel', 'ssvkernel',
           'sshist_classic', 'sskernel_classic', 'ssvkernel_classic')
