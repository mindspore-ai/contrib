from setuptools import setup, find_packages

setup(
  name = 'nystrom-attention',
  packages = find_packages(),
  version = '0.0.12',
  license='MIT',
  description = 'Nystrom Attention - Pytorch',
  long_description_content_type = 'text/markdown',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/nystrom-attention',
  keywords = [
    'artificial intelligence',
    'attention mechanism'
  ],
  install_requires=[
    'mindspore>=2.3',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)

