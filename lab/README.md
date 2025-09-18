# LLM and Societal Consequences of AI - Lab and Assignment

This is the boilerplate code and exercises of the lab and assignment of the
course "Large Language Models and Societal Consequences of Artificial Intelligence"
(1RT730) at Uppsala University.

Feel free to look and try these exercises before-hand, attending the lab is optional and is allocated to help students with problems or questions.

The goal of the lab sessions is to provide hands-on experience with the tools and methods necessary to build an AI chatbot using pretrained models locally or in the cloud.

There are two main parts to the lab sessions:
1. Getting acquainted with the AI models API, it has two path (choose one):

   a. Using Google's Gemini API in the cloud.

   b. Using the HuggingFace API on your machine (requires a decent hardware setup).

Using the knowledge gained through the lab exercises you will then complete the assignment: 
* Building a chatbot using the results from the previous step. The chatbot is a mushroom expert that can answer questions about mushrooms, and work with images.

## Installation

The code runs on Python (3.12), there are three ways you can proceed with the installation:
1. Install docker and use the provided docker compose configuration.
2. Install Python with mamba/conda and install the provided environment.txt files.
3. Install Python and install the required packages manually.

We recommend option 1 as it ensures you have that same exact environment that this lab was tested on. This lab was also tested on a virtual environment with pip but small changes in environment can still cause some issues.

The instructions are detailed in [installation.md](installation.md).

## Lab Exercises

Complete one of the notebooks depending on the path you choose. The notebooks have exercises that you can complete to get a better understanding of the API and the models. They give you a good starting point for building the chatbot for the assignment.

It is advised to complete the exercises of the lab before moving to the assignment.

### Option a.: Using the Gemini API

The notebook `gemini.ipynb` contains all the exercises for the Gemini API. Just start a Jupyter notebook server and open the notebook to start working on the exercises.

### Option b.: Using the HuggingFace API

The notebook `huggingface.ipynb` contains all the exercises for the HuggingFace API. Just start a Jupyter notebook server and open the notebook to start working on the exercises.

You **do not** need to submit the completed code of the lab. The lab is just a learning experience to get you started before the assignment(s) and the project.
