from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import find_packages, setup

setup_args = generate_distutils_setup(
    packages=find_packages(),
    package_dir={"": ""},
)
setup(**setup_args)
