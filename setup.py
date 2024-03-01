from setuptools import setup

setup(name='py_portada_image',
    version='0.1',
    description='tools for processing images within the PortADa project',
    author='PortADa team',
    author_email='jcbportada@gmail.com',
    license='MIT',
    packages=['py_portada_image'],
    install_requires=[ 
        'scikit-image',
        'deskew',
        'numpy'
    ], 
    zip_safe=False)
