import setuptools

setuptools.setup(
    name="mkidresonatorkal",
    version="0.1",
    author="MazinLab",
    author_email="mazinlab@ucsb.edu",
    description="An UVOIR MKID ML Calibration Package",
    long_description_content_type="text/markdown",
    url="https://github.com/mailesasaki/mkidresonatorkal",
    scripts=['mkidresonatorkal/findResonatorsWPS.py'],
    packages=setuptools.find_packages(),
)

