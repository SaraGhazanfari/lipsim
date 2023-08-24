import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
long_description = "".join(long_description.split("<!--Experiments-->")[::2])
long_description = "".join(long_description.split("![teaser](images/figs/teaser.png)"))

setuptools.setup(
    name="lipsim",
    version="0.1.0",
    description="LipSim similarity metric",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SaraGhazanfari/lipsim",
    packages=['lipsim', 'lipsim/core'],
    install_requires=[
        "numpy",
        "open-clip-torch",
        "peft==0.1.0",
        "Pillow",
        "torch",
        "timm",
        "scipy",
        "torchvision",
        "transformers"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
