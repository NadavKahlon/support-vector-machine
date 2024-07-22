from setuptools import setup, find_packages

setup(
    name='support-vector-machine',
    packages=find_packages(include=['svm']),
    install_requires=[
        'numpy'
    ],
    extras_require={
        'examples': ['scikit-learn']
    },
    include_package_data=True,
)
