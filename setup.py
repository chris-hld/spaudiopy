import setuptools

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setuptools.setup(name='spaudiopy',
                 version='0.1.1',
                 description='Spatial Audio Python Package',
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 url='https://github.com/chris-hld/spaudiopy',
                 author='Chris Hold',
                 author_email='Chris.Hold@mailbox.tu-berlin.de',
                 license='MIT',
                 packages=setuptools.find_packages(),
                 package_data={'spaudiopy': ['t_designs_1_21.mat',
                                             'fliegeMaierNodes_1_30.mat'
                                             ]},
                 install_requires=[
                                   'numpy',
                                   'scipy',
                                   'pandas',
                                   'joblib',
                                   'matplotlib',
                                   'soundfile',
                                   'sounddevice',
                                   'resampy',
                                   'h5py',
                                   'quadpy'
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
