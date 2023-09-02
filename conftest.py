import pytest

import stac
@pytest.fixture(autouse=True)
def add_doctest_imports(doctest_namespace):
    doctest_namespace['stac'] = stac
