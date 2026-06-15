"""Shared fixtures for the end-to-end persona-pipeline tests.

The ``ovos-persona`` wheels published to PyPI do not ship the
``ovos_persona/locale`` directory (the locale resources are declared only in
``MANIFEST.in``, which affects the sdist but not the wheel).  When that
directory is absent, ``ovos_utils.lang.get_language_dir`` raises
``FileNotFoundError`` while the persona pipeline plugin loads its intent
samples, which aborts the whole plugin and leaves the persona pipeline stages
unavailable.

This fixture restores a usable locale directory for the *installed*
``ovos_persona`` package before any MiniCroft is started, so the real pipeline
can load and the end-to-end behaviour can be exercised.  Locale files bundled
in the installed package are preserved; only the missing base directory (and an
``en-US`` placeholder) is created when needed.
"""
import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _ensure_persona_locale():
    """Guarantee the installed ovos_persona package exposes a locale directory.

    Works around a packaging defect in published ovos-persona wheels where the
    locale resources are missing, causing the persona pipeline plugin to crash
    on load.  Creating the base locale directory is enough for the plugin to
    load cleanly; intent matching is not what these end-to-end tests assert.
    """
    try:
        import ovos_persona
    except ImportError:
        # ovoscope / persona not installed; the test module skips itself.
        yield
        return

    locale_root = os.path.join(os.path.dirname(ovos_persona.__file__), "locale")
    if not os.path.isdir(locale_root):
        os.makedirs(os.path.join(locale_root, "en-US"), exist_ok=True)

    yield
