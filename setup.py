from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='py_portada_image',
    version='0.1.6',
    description='tools for processing images within the PortADa project',
    author='PortADa team',
    author_email='jcbportada@gmail.com',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
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
