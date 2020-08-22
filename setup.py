from setuptools import setup, find_packages

setup(
    name="miceForest",
    version="1.0.1",
    author="Samuel Wilson",
    license="MIT",
    author_email="samwilson303@gmail.com",
    description="Perform MICE",
    long_description="Perform MICE with Random Forests",
    url="https://github.com/AnotherSamWilson/miceForest",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
