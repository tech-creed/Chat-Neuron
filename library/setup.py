from setuptools import setup, find_packages

setup(
    name='ChatNeuron',
    version='0.1',
    license='MIT',
    author="Tech Creed",
    author_email='techcreed.tech@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/tech-creed/Chat-Neuron',
    keywords='Deep Learning Trainable AI Chatbot',
    install_requires=[
          'scikit-learn',
      ],

)