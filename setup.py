"""For pip install."""

import setuptools
from os import path


# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setuptools.setup(name='spaudiopy',
                 version_config={
                    "template": "{tag}",
                    "dev_template": "{tag}.post{ccount}+git.{sha}",
                    "dirty_template": "{tag}.post{ccount}+git.{sha}.dirty",
                    "starting_version": "0.0.0",
                 },
                 setup_requires=['setuptools-git-versioning'],
                 description='Spatial Audio Python Package',
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 url='https://github.com/chris-hld/spaudiopy',
                 author='Chris Hold',
                 author_email='Chris.Hold@mailbox.tu-berlin.de',
                 license='MIT',
                 packages=setuptools.find_packages(),
                 package_data={'spaudiopy': ['../data/Grids/*.mat',
                                             '../data/ls_layouts/*.json'
                                             ],
                               },
                 install_requires=[
                                   'numpy',
                                   'scipy',
                                   'pandas',
                                   'joblib',
                                   'matplotlib !=3.1.*, !=3.2.*',  # axis3d aspect broken
                                   'soundfile',
                                   'sounddevice',
                                   'resampy',
                                   'h5py'
                                   ],
                 platforms='any',
                 python_requires='>=3.6',
                 classifiers=[
                     "Development Status :: 3 - Alpha",
                     "License :: OSI Approved :: MIT License",
                     "Operating System :: OS Independent",
                     "Programming Language :: Python",
                     "Programming Language :: Python :: 3",
                     "Programming Language :: Python :: 3 :: Only",
                     "Topic :: Scientific/Engineering",
                 ],
                 zip_safe=True,
                 )
