import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bayne",
    version="0.0.1",
    author="Frederik Baymler Mathiesen",
    author_email="frederik@baymler.com",
    description="Bayesian Neural Networks in Pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zinoex/bayne",
    project_urls={
        "Bug Tracker": "https://github.com/zinoex/bayne/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "bayne"},
    packages=setuptools.find_packages(where="bayne"),
    python_requires=">=3.6",
)
