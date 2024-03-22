from setuptools import setup

setup(name='py_portada_image',
    version='0.1.5',
    description='tools for processing images within the PortADa project',
    author='PortADa team',
    author_email='jcbportada@gmail.com',
    license='MIT',
    url="https://github.com/portada-git/py_portada_image",
    packages=['py_portada_image'],
    py_modules=['deskew_tools','dewarp_tools'],
    install_requires=[ 
        'scikit-image',
        'deskew',
        'numpy >= 1.21,<2',
        'pillow',
        'scipy',
        'opencv-python >= 4.8,<4.9'
    ], 
    python_requires='>=3.9',
    zip_safe=False)
