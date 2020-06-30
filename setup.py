import setuptools
from os import path

__version__ = "unknown"


# "import" __version__
for line in open("spaudiopy/__init__.py"):
    if line.startswith("__version__"):
        exec(line)
        break

# read the contents of your README file
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()



setuptools.setup(name='spaudiopy',
                 version=__version__,
                 description='Spatial Audio Python Package',
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 url='https://github.com/chris-hld/spaudiopy',
                 author='Chris Hold',
                 author_email='Chris.Hold@mailbox.tu-berlin.de',
                 license='MIT',
                 packages=setuptools.find_packages(),
                 package_data={'spaudiopy': ['t_designs_1_21.mat',
                                             'n_designs_1_124.mat',
                                             'fliegeMaierNodes_1_30.mat',
                                             'lebedevQuadratures_3_131.mat',
                                             '../data/ls_layouts/*.json'
                                             ],
                               },
                 install_requires=[
                                   'numpy',
                                   'scipy',
                                   'pandas',
                                   'joblib',
                                   'matplotlib<3.1.0',  # https://github.com/matplotlib/matplotlib/issues/1077
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
                     "Programming Language :: Python :: 3.6",
                     "Programming Language :: Python :: 3.7",
                     "Programming Language :: Python :: 3 :: Only",
                     "Topic :: Scientific/Engineering",
                 ],
                 zip_safe=True,
                 )
