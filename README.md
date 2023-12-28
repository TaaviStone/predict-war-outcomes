# Prediction model for outcomes of wars

Authors - Taavi Raudkivi, Karl Mumme

----

## Background

### Project motivation

Since we're both volunteers at Estonian Defence League and are interested in decision-making and leading at a higher level than 
a squad, we would like to see what factors contribute to winning and losing based on historical data.

### Project focus
Our project focuses on developing a predictive model to assist the military in strategic decision-making. The primary objective is to create a tool that enables the simulation of various war scenarios, helping military planners understand the potential outcomes based on
different combinations of assets and factors.

### Goals
There were three main business goals: 
1. Enhance Strategic Planning - provide a sophisticated tool for military strategists to simulate and analyze different war scenarios.
2. Optimize resource Allocation - assist in determining the most effective deployment of assets to maximize the chances of success in a conflict.
3. Improve Decision-Making - support military decision-makers with data-driven insights
to make informed choices during planning and ongoing conflicts.

### Success Criteria

1. Simulated scenario success rate - we will measure success by the model's ability to predict outcomes in simulated scenarios, with the goal of surpassing historical success rates.
2. Resource allocation efficiency - success will be demonstrated through a tangible reduction in resource allocation inefficiencies, leading to optimized deployment strategies.
3. Real-world conflict outcomes - ultimately, success will be evident in the real-world application of the model, where enhanced decision-making contributes to successful outcomes in
military conflicts.

### Data mining goals

1. Predictive Modeling for War Outcomes - develop a sophisticated predictive model leveraging historical data and relevant environmental variables to discern and forecast the outcomes of wars accurately.
2. Scenario Simulation Capabilities - enable the military to simulate diverse war scenarios, providing valuable insights into potential outcomes under varying conditions. This includes creating a tool that enhances strategic planning by considering a spectrum of potential scenarios.
3. Optimization of Resource Allocation - assist military planners in optimizing the allocation of assets by identifying the most effective combinations for different scenarios, ultimately
reducing inefficiencies and enhancing overall strategic effectiveness.

----

## Project setup and how to run it

The project repository is set into 4 different directories of which 3 contain necessary python files and one is the directory for the data.
The necessary directories are:
* [Setup](https://github.com/TaaviStone/predict-war-outcomes/tree/main/setup)
* [Models](https://github.com/TaaviStone/predict-war-outcomes/tree/main/Models)
* [Graphs](https://github.com/TaaviStone/predict-war-outcomes/tree/main/Graphs)


Setup directory has a file DataSetup.py which does the initial data cleaning and setup process in order to create models and graphs.

Models directory has all the necessary python files that generate the prediction models and produce the graphs to accompany the models.

Graphs directory has some data files for the graphs and a InformationalGraphs.py file which generates the graphs after running it. 
All the graphs except for the ones that measure the models have come from that specific python file.

In order to get the same results, on must:
1. Run the DataSetup.py
2. Run the python files under the Models directory
3. Run the InformationalGraphs.py

----
## Data overview

### Data origin
Our data comes from kaggle and all the files are available on [here](https://www.kaggle.com/datasets/residentmario/database-of-battles).

## Data description

### 0 : Defender
### 1 : Attacker

### Wina Reference
* -1 : attacker lost
* 0 : draw
* 1 : defender lost

### Element of surprise reference
* -3,-2, -1 : surprise achieved by the defender (-3: most, -2: substantial , -1: minor)
* 0 : neither side had the element of surprise
* 1, 2, 3 : surprise achieved by the attacker (3: most, 2: substantial , 1: minor)

### Terrain reference
#### Terra1
* 0 : None
* R : Rolling
* G : Rugged
* F : Flat

#### Terra2
* 0 : None
* B : Bare
* M : Mixed
* D : Desert
* W : Heavily wooded

#### Terra3
* 0 : None
* M : Marsh or Swamp
* U : Mixed
* D : Dunes

### Weather reference
#### wx1
* D : Dry
* W : Wet

#### wx2
* H : Heavy Precipitation
* S : Sunny (no precipitation)
* L : Light Precipitation
* O : Overcast (no precipitation)

#### wx3
* H : Hot
* C : Cold
* T : Temparate

#### wx4
* S : Summer
* $ : Spring
* W : Winter
* F : Fall

#### wx5
* E : Tropical
* D : Desert
* T : Temparate


