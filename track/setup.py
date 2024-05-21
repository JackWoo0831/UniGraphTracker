from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(['models/association_graph.py', 'models/frame_graph.py',
                            'models/graph_net.py', 'models/loss.py', 'models/uni_graph.py'])
)
