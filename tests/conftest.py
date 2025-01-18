import pytest


@pytest.fixture
def username(request):
    """
    Provides the username from command-line to tests that accept a `username` fixture.
    """
    return request.config.getoption("--username")


@pytest.fixture
def password(request):
    """
    Provides the password from command-line to tests that accept a `password` fixture.
    """
    return request.config.getoption("--password")


def pytest_addoption(parser):
    """
    Add command-line options for the OpenReview username and password.
    """
    parser.addoption("--username", action="store", default=None, help="OpenReview username")
    parser.addoption("--password", action="store", default=None, help="OpenReview password")
