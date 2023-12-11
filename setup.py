import setuptools
import glob
import os

fname = 'requirements.txt'
with open(fname, 'r', encoding='utf-8') as f:
	requirements =  f.read().splitlines()

required = []
dependency_links = []

# Do not add to required lines pointing to Git repositories
EGG_MARK = '#egg='
for line in requirements:
	if line.startswith('-e git:') or line.startswith('-e git+') or \
		line.startswith('git:') or line.startswith('git+'):
		line = line.lstrip('-e ')  # in case that is using "-e"
		if EGG_MARK in line:
			package_name = line[line.find(EGG_MARK) + len(EGG_MARK):]
			repository = line[:line.find(EGG_MARK)]
			required.append('%s @ %s' % (package_name, repository))
			dependency_links.append(line)
		else:
			print('Dependency to a git repository should have the format:')
			print('git+ssh://git@github.com/xxxxx/xxxxxx#egg=package_name')
	else:
		if line.startswith('_'):
			continue
		required.append(line)
setuptools.setup(
     name='sonata',
     use_scm_version=True,
     setup_requires=['setuptools_scm'],
     packages=['sonata'],
     package_dir={'': 'src'},
     py_modules=["sonata"+'.'+os.path.splitext(os.path.basename(path))[0] for path in glob.glob('src/sonata/*.py')],
     install_requires=required,
     dependency_links=dependency_links,
     author="XXX",
     author_email="xxx@example.com",
     description="your_project_description",
     long_description=open('README.md').read(),
     url="https://github.com/DoraDong-2023/SONATA",
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ]
 )
