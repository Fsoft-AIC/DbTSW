from setuptools import setup, find_packages

setup(
    name='db_tsw',
    version='0.1',
    packages=find_packages(include=['db_tsw']),
    install_requires=[
        'torch',
    ],
    author='Khoi N.M. Nguyen',
    author_email='minhkhoi1026@gmail.com',
    description='Distance-Based Tree-Sliced Wasserstein Distance',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Fsoft-AIC/DbTSW',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
