[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=8127826&assignment_repo_type=AssignmentRepo)
<!--
Name of your teams' final project
-->
# final-project
## [National Action Council for Minorities in Engineering(NACME)](https://www.nacme.org) Google Applied Machine Learning Intensive (AMLI) at the `UNIVERSITY OF ARKANSAS`

<!--
List all of the members who developed the project and
link to each members respective GitHub profile
-->
Developed by: 
- [Santiago Dorado](https://github.com/dorasanti) - `UNIVERSITY OF ARKANSAS`
- [Devin Hill](https://github.com/Vuxify) - `UNIVERSITY OF ARKANSAS` 
- [Ayia Ismael](https://github.com/daholypandah) - `VIRGINIA TECH` 
- [Adrian Whitty](https://github.com/adrianwhitty2022) - `UNIVERSITY OF ARKANSAS-PINE BLUFF`

## Objective
The objective of this project is to use machine learning models to predict the location of a aerial drone. These models rely on the real-time high-altitude to maintain geolocation of the drone.
## Goal
Long term
    - To use Convolutional Neural Networks in application to Absolution Visual Geolocation to prevent the possible drawbacks and lack of secure navigation when navigating by GNSS. In the long run, taking this model and its subsequent algorithmic modeling and applying it to sections outside of the Washington County area that the current images used for testing and training are based on. 
Short term
    - Optimize the models for better accuarcy uses 12 training set and 1 testing set for Washington County.

## Describe the dataset, data acquisition, and data preparation
- Dataset:  Agricultural land images taken in Washington county from 2006 to 2020.

- Data Acquisition: The data we acquired were developed and constructed previously by Dr. Rainwater and Winthrop Harvey.
    -
- Data Preparation: Data gathering, data transformation and data validation were all methods we used to prepare our data.

## Models
 For this project, we decided to use Xception and Vision Transformer models. The models that we used were developed and constructed previously by Dr. Rainwater and Winthrop Harvey. Our main objective was to improve the performance of these models. 
 ViT
 The Vision transformer is based on the structure of a transformer designed for text-based tasks. 
 ![Picture1](https://user-images.githubusercontent.com/106926413/181702423-74f5c63d-6548-43e1-9e42-90a9a4d9bf75.png)
 Xception
 Xception is a convolutional neural network. The pretrain Xception network can classify 1000 object categories, but in our class we are using Xception to categorize location of images.

 ![Picture2](https://user-images.githubusercontent.com/106926413/181702662-bf04c9ea-745d-40cb-8d95-4e692f2a5fa5.jpg)
 
## Results
Based on our model’s performance, we have deemed that the model that utilizes Vision Transformers was the best performing model. This was due to the optimal RMSE score and loss score being less than that of the model using Xception. Although Xception did have slightly higher accuracy, vision transformers are optimal for large datasets that include millions of images. Since we only had a dataset of 24,000 images, the accuracy, and performance of the model using vision transformers underperformed due to the lack of data. This however is not a real issue since our dataset was only collected from an area of thirty kilometers squared. 

## Usage instructions
<!--
Give details on how to install fork and install your project. You can get all of the python dependencies for your project by typing `pip3 freeze requirements.txt` on the system that runs your project. Add the generated `requirements.txt` to this repo.
-->
1. Fork this repo
2. Change directories into your project
3. On the command line, type `pip3 install requirements.txt`
4. ....
