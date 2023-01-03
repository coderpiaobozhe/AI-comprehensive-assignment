from setuptools import setup

setup(
    name="classifier_guided-diffusion",
    py_modules=["classifier_guided_diffusion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
