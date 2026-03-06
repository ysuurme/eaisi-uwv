"""
Purpose:    Init file - Utilities UWV project, similar to utils_pieter.
"""

# Set version number
__version__ = "1.0.0"

# Import modules - By using this approach we can import the functions
# in to the Jupyter Notebook and there is no need to refer to the module
# name. In the Jupyter Notebook we can write:
# "from utils import f_info"
# and use the f_info function in the Jupyter Notebook.

# In case we exclude the lines below, we have to state:
# "from utils.f_info import f_info"
# in order to use said function in the Jupyter Notebook.

# There is no need use the same name for the function and the module
# (Python file). Here, I have use one module per function.

# If you have multiple functions that are related you can put them
# in the same module. Below you could write, in 
# from .module_name import f_one
# from .module_name import f_two
# Or in your Jupyter Notebook, you could write:
# from utils.module_name import f_one
# from utils.module_name import f_two

from .m_nb_results_to_gold_export import f_nb_results_to_gold_export
from .m_nb_results_to_gold_export import f_list_gold_tables
from .m_query_database import f_query_database