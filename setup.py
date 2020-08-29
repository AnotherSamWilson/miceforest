from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="miceforest",
    version="1.0.5",
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
    url="https://github.com/AnotherSamWilson/miceforest",
    packages=find_packages(),
    classifiers=[
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
