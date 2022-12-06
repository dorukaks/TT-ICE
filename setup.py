from setuptools import setup

setup(
    name="DaMAT",
    version="0.1.0",
    description="An initial module for Tensor Train decomposition.",
    license="GNU",
    author="Doruk Aksoy",
    author_email="doruk@umich.edu",
    packages=["DaMAT"],
    install_requires=["numpy", "datetime", "Jinja2", "pdoc", "Pygments", "MarkupSafe"],
)
