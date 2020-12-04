from setuptools import setup

setup(
    name='pureples',
    version='0.0',
    author='adrian, simon',
    author_email='mail@adrianwesth.dk',
    maintainer='simon, adrian',
    maintainer_email='mail@adrianwesth.dk',
    url='https://github.com/ukuleleplayer/pureples',
    license="MIT",
    description='HyperNEAT and ES-HyperNEAT implemented in pure Python',
    long_description='Python implementation of HyperNEAT and ES-HyperNEAT ' +
                     'developed by Adrian Westh and Simon Krabbe Munck for evolving arbitrary neural networks. ' +
                     'HyperNEAT and ES-HyperNEAT is originally developed by Kenneth O. Stanley and Sebastian Risi',
    packages=['pureples', 'pureples/hyperneat', 'pureples/es_hyperneat', 'pureples/shared'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.x',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering'
    ], 
    install_requires=['numpy', 'neat-python', 'graphviz', 'matplotlib', 'gym']
)
