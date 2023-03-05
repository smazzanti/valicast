from setuptools import setup

setup(
    name='valicast',
    version='0.1',
    description='Validation methods for forecasts',
    url='https://github.com/smazzanti/valicast',
    author='Samuele Mazzanti',
    author_email='mazzanti.sam@gmail.com',
    license='MIT',
    packages=['src'],
    install_requires=[
        'numpy>=1.18.1'
    ],
    zip_safe=False
)
