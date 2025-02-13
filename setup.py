from setuptools import setup, find_packages

setup(
    name="segmenting_living_organisms",
    version="0.1",
    packages=find_packages(include=["torchtmpl", "torchtmpl.*"]),
)

