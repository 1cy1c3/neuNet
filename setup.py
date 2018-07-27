from setuptools import setup

setup(
    name='neuNet',
    version='1.0.0',
    packages=['com', 'com.runekrauss', 'com.runekrauss.neunet', 'com.runekrauss.neunet_test'],
    url='https://runekrauss.com',
    license='GNU GENERAL PUBLIC LICENSE',
    author='Rune Krauss',
    author_email='contact@runekrauss.com',
    description='NeuNet represents an artificial neuronal network that is a mathematical replica of neurons in our brain.',
    install_requires=['matplotlib', 'scikit-learn']
)
