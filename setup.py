from setuptools import setup, find_packages

setup(
    name="chat2table",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "python-dotenv>=1.1.0",
        "openai>=1.76.0",
        "Pillow>=11.2.1",
        "requests>=2.32.3",
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "langchain-openai>=0.0.5",
        "pdf2image==1.16.3",
    ],
    entry_points={
        "console_scripts": [
            "chat2table=src.main:main",
        ],
    },
) 