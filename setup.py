from setuptools import setup, find_packages

setup(
    name="qtom",
    version="1.0.0",
    description="Neural Network Quantum State Tomography Library",
    author="Manuel A. Garcia & Johan Garzon",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "tensorflow>=2.5.0",
        "keras>=2.5.0",
    ],
    python_requires=">=3.7",
    entry_points={
        'console_scripts': [
            'qtnn-train=quantum_tomography_nn.cli.train:main',
            'qtnn-evaluate=quantum_tomography_nn.cli.evaluate:main',
            'qtnn-generate-data=quantum_tomography_nn.cli.generate_data:main',
        ],
    },
    keywords="quantum, tomography, neural networks, machine learning",
)
