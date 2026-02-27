from setuptools import setup

setup(
  name    = "adaptivekde",
  packages = [ "adaptivekde" ],
  version = "1.2.0",
  description = 'Optimal fixed or locally adaptive kernel density estimation',
  long_description=open('README.txt').read(),
  long_description_content_type='text/markdown',
  author = "Lee A.D. Cooper",
  author_email = 'cooperle@gmail.com',
  url = 'https://github.com/shimazaki/AdaptiveKDE',
  python_requires='>=3.9',
  install_requires=['numpy>=1.24'],
  classifiers = ['Development Status :: 5 - Production/Stable',
          'Environment :: Console',
          'License :: OSI Approved :: Apache Software License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Software Development :: Libraries :: Python Modules'],
)
