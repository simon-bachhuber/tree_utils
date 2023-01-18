import setuptools

setuptools.setup(
    name="tree_utils",
    packages=setuptools.find_packages(),
    version="0.1.0",
    install_requires=["jax", "jaxlib", "tree"],
)
