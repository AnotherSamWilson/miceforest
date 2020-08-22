from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="miceForest",
    version="1.0.1",
    author="Samuel Wilson",
    license="MIT",
    author_email="samwilson303@gmail.com",
    description="Perform MICE",
    keywords=['MICE','Imputation','Missing Values','Missing','Random Forest'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['sklearn',
                      'numpy',
                      'pandas',
                      'seaborn',
                      'matplotlib'
                      ],
    url="https://github.com/AnotherSamWilson/miceForest",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
