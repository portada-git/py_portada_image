from setuptools import setup

setup(name='py_portada_image',
    version='0.1.0',
    description='tools for processing images within the PortADa project',
    author='PortADa team',
    author_email='jcbportada@gmail.com',
    license='MIT',
    url="https://github.com/portada-git/py_portada_image",
    packages=['py_portada_image'],
    py_modules=['deskew_tools'],
    install_requires=[ 
        'scikit-image',
        'deskew',
        'numpy'
    ], 
    python_requires='>=3.9',
    zip_safe=False)
