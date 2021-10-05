from setuptools import setup
with open('requirements.txt', 'r') as f:
    reqs = f.read()
with open('LICENSE', 'r') as f:
    legal = f.read()
with open('README.md', 'r') as f:
    readme = f.read()
setup(
    name='reading4listeners',
    version='0.0.2',
    packages=['r4l'],
    url='https://github.com/CypherousSkies/reading-for-listeners',
    #license=legal,
    license='AGPL-3',
    author='CypherousSkies',
    author_email="5472563+CypherousSkies@users.noreply.github.com",
    description='A deep-learning powered application which turns pdfs into audio files. Featuring ocr improvement and tts with inflection!',
    #long_description=readme,
    install_requires=reqs,
    entry_points={"console_scripts": ["r4l = r4l.bin.cli:main"], }
)
