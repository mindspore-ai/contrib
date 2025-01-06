from setuptools import setup, find_packages

setup(
  name = 'htm-pytorch',
  packages = find_packages(),
  version = '0.0.4',
  license='MIT',
  description = 'Hierarchical Transformer Memory - mindspore',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/htm-pytorch',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'attention-mechanism',
    'memory'
  ],
  install_requires=[
    'einops>=0.3',
    'mindspore=2.2.14'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
