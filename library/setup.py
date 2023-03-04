from setuptools import setup, find_packages

setup(
    name='ChatNeuron',
    version='1.0',
    license='MIT',
    author="Tech Creed",
    author_email='techcreed.tech@gmail.com',
    packages=find_packages('ChatNeuron'),
    package_dir={'': 'ChatNeuron'},
    url='https://github.com/tech-creed/Chat-Neuron',
    keywords='Deep Learning Trainable AI Chatbot',
    install_requires=[
          'pandas',
          'numpy',
          'tensorflow'
      ],

)