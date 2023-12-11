from setuptools import setup, find_packages

setup(
    name='deasc',
    version='0.1',
    packages=find_packages(),
    author='Filippo Gori',
    author_email='f.gori21@imperial.ac.uk',
    description='A data-integrated framework for wake steering control of wind farms.',
    long_description='A data-integrated framework for wake steering control of wind farms.',
    long_description_content_type='text/markdown',
    url='https://github.com/fg320/DEASC',
    install_requires=[
        'numpy>=1.23.5',
        'GPy>=1.10.0',
        'floris>=3.4',
        'TuRBO>=0.0.1',
        'multiprocess>=0.70.15'
    ],
    license='Apache-2.0',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
    ],
)
