# Generate Sphinx documentation (codedoc)
/!\ In construction

To generate the documentation, you will need Sphinx, a Documentation building tool, and a nice-looking custom [Sphinx theme similar to the one of readthedocs.io](https://sphinx-rtd-theme.readthedocs.io/en/latest/):
```
pip install sphinx sphinx_rtd_theme recommonmark
```
Then in the current folder:
```
sphinx-build -b html ./source ./build
```
The html will be available within the folder doc/build.

# Files
The project_introduction.pdf file is an introduction to the subject of this software. The document is a bit outdated regarding some of the modelization choices, howerver the overall motivations and goals are identical.
