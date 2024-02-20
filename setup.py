from setuptools import setup, find_packages
from os import path
import versioneer

cur_dir = path.abspath(path.dirname(__file__))

# parse requirements
with open(path.join(cur_dir, "requirements.txt"), "r") as f:
    requirements = f.read().split()

setup(
    name="torch_model",
    author="Jesse Bellister",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    #  license="...",
    install_requires=requirements,
    url="https://github.com/jbellister-slac/torch-model.git",
    include_package_data=True,
    python_requires=">=3.8",
    entry_points={
        "orchestration": [
            "torch_model.model=\
                torch_model.model:TorchModel",
            "torch_model.flow=\
                torch_model.flow:flow",
        ]
    },
)
