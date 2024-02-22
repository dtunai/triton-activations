from setuptools import setup, find_packages

exec(open("triton_activations/__version__.py").read())

with open("README.md", "r", encoding="utf-8") as readme_file:
    long_description = readme_file.read()

with open("requirements.txt", "r") as requirements_file:
    install_requires = requirements_file.read().splitlines()

setup(
    name="triton-activations",
    version=VERSION,
    description="A expanded collection of Neural Network activations and other functions for Triton Compiler",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="simudt",
    author_email="dogukanuraztuna@gmail.com",
    url="https://github.com/simudt/triton_activations",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    keywords="triton activations library",
    install_requires=install_requires,
    python_requires=">=3.10.12",
)
