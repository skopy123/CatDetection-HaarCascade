import pytest

from catalarm.config import models_dir
from catalarm.tools.selftest import run

from .conftest import TEMPLATES, TEST_IMG


@pytest.mark.models
def test_real_models_identify_test_photos():
    assert run(TEST_IMG, TEMPLATES, models_dir()) == []
