import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="humanmotionanalysis",
    version="0.0.1",
    author="IISY at Beuth",
    author_email="iiysy@beuth-hochschule.de",
    description="Human Motion Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.beuth-hochschule.de/iisy/humanmotionanalysis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)