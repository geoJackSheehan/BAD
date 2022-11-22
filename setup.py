import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name='bad_package',
      version='1.0',
      packages= setuptools.find_packages(),
      author="Annabel Yim, Jack Sheehan, Danhee Kim, Hope Neveux",
      description="The bad package used for automatic differentiation",
      long_description= long_description,
      long_description_content_type="text/markdown",
      url="https://code.harvard.edu/CS107/team23",
      packages = setuptools.find_packages(),
      classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
      ],
      python_requires='>=3.6',
      install_requires = [ 'numpy', 'pytest' ]
)
