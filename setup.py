import setuptools

with open("README.md", "r",  encoding="utf-8") as fh:
    long_description = fh.read()

REPO_NAME = "chicken-disease-classification"
AUTHOR_USER_NAME = "brijesh24bs"
SRC_REPO = "CNNClassifier"

setuptools.setup(
    name=SRC_REPO,
    version="0.0.1",
    author=AUTHOR_USER_NAME,
    description="a small project on chicken disease classification using CNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    author_email="brijesh24.bs@gmail.com",

    package_dir={"": "src"},
    packages=setuptools.find_packages(where='src'),

    license="MIT",
    python_requires=">=3.6",
)