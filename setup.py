from setuptools import setup, find_packages

setup(
    name="lipsim",
    version="0.1.0",
    description="LipSim similarity metric",
    url="https://github.com/SaraGhazanfari/lipsim",
    packages=find_packages(include=['lipsim', 'lipsim.*']),
    install_requires=[
        "dreamsim",
        "pytorch-warmup",
        "torchvision",
        "pandas",
    ],
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    # ],
)
