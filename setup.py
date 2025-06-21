# setup.py
from setuptools import setup, find_packages

setup(
    name="dreams-data-science",
    version="0.1.0",
    py_modules=["utils"],  # Individual .py files
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
)