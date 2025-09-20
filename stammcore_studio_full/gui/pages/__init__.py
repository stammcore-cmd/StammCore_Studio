"""Pages for the StammCore Studio Full GUI."""

from .dummy_generator_page import DummyGeneratorPage  # noqa: F401
from .training_page import TrainingPage  # noqa: F401
from .library_page import LibraryPage  # noqa: F401
from .test_page import TestPage  # noqa: F401

__all__ = ["DummyGeneratorPage", "TrainingPage", "LibraryPage", "TestPage"]