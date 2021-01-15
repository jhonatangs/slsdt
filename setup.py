import io
import os
import sys
from shutil import rmtree

from setuptools import setup, Command, find_packages

NAME = "slsdt"
DESCRIPTION = "Oblique decision tree using the LAHC heuristic. "
URL = "https://github.com/jhonatangs/slsdt"
EMAIL = "jhonatan.souza@aluno.ufop.edu.br"
AUTHORS = "Souza, J.G. and Santos, H.G."
REQUIRES_PYTHON = ">3.5.0"
VERSION = "0.0.2"

REQUIRED = ["numpy", "pandas", "numba", "scikit-learn"]

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

about = {}
try:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
except FileNotFoundError:
    about["__version__"] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(about["__version__"]))
        os.system("git push --tags")

        sys.exit()


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=[
        "Oblique Decision Tree",
        "Machine Learning",
        "Classification",
        "Heuristic",
        "Optimization",
    ],
    author=AUTHORS,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=REQUIRED,
    include_package_data=True,
    license="EPL-2.0",
    classifiers=[
        "License :: OSI Approved :: Eclipse Public License 2.0 (EPL-2.0)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
    ],
    # $ setup.py publish support.
    cmdclass={
        "upload": UploadCommand,
    },
)