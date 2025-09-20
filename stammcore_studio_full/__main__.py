"""Package entry point for StammCore Studio Full.

This file enables launching the application via ``python -m stammcore_studio_full``.
It simply imports and executes the ``main`` function from ``main.py``.  Having
this module avoids relative import errors when the package is executed as a
module rather than via direct script invocation.

Usage
-----

To launch the GUI from a source checkout or installed package run:

.. code-block:: bash

    python -m stammcore_studio_full

If you prefer to run the module directly you can still call
``python main.py`` inside the package directory.
"""

from .main import main


if __name__ == "__main__":
    main()