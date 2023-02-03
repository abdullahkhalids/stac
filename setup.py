from setuptools import setup

from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="stacdev",
    version="0.1",
    author="Abdullah Khalid",
    author_email="abdullah@abdullahkhalid.com",
    description="A python library to play with stabilizer codes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abdullahkhalids/stac",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["ipython",
                      "numpy",
                      "qiskit",
                      "stim",
                      "tabulate",
                      "svg.py"]
)
