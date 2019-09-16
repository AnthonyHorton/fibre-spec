import setuptools

with open("README.md",  "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fibrespec",
    version="0.1",
    author="Anthony Horton",
    author_email="anthony.horton@mq.edu.au",
    description="Tools for quick-look analysis of optical fibre fed spectrograph data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AnthonyHorton/fibre-spec",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Licence :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Astronomy"
    ],
)
