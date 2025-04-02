from setuptools import setup, find_packages

setup(
  name = 'nwt-mindspore',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'NWT - MindSpore',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'mindspore',
    'audio to video synthesis'
  ],
  install_requires=[
    'numpy',
    'mindspore'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9',
  ],
)