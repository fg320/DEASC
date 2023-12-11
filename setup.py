from setuptools import setup, find_packages

setup(
    name='deasc',
    version='0.1',
    packages=find_packages(),
    author='Filippo Gori',
    author_email='f.gori21@imperial.ac.uk',
    description='A data-integrated framework for wake steering control of wind farms.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/fg320/DEASC',
    install_requires=[
        'GPy>=1.10.0'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
)
