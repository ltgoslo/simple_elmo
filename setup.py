import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simple_elmo",
    version="0.1.0",
    author="Andrey Kutuzov",
    author_email="andreku@ifi.uio.no",
    description="Useful library to work with pre-trained ELMo embeddings in TensorFlow ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ltgoslo/simple_elmo",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=["tensorflow>1.15", "h5py", "numpy", "smart_open>1.8.1", "pandas", "scikit-learn"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Utilities"
    ],
)
