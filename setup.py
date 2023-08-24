from setuptools import setup, find_packages

setup(
    name="lipsim",
    version="0.1.0",
    description="LipSim similarity metric",
    url="https://github.com/SaraGhazanfari/lipsim",
    packages=find_packages()#['lipsim', 'lipsim/core'],
    # install_requires=[
    #     "numpy",
    #     "open-clip-torch",
    #     "peft==0.1.0",
    #     "Pillow",
    #     "torch",
    #     "timm",
    #     "scipy",
    #     "torchvision",
    #     "transformers"
    # ],
    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    # ],
)
