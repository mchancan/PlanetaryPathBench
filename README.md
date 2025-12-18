# Create conda environment

	conda env create -f env.yml

# Datasets

Download the [MoonPlanBench](https://drive.google.com/drive/folders/15srtIABvwBSbILQESVvAFPHMc3TCzS_R?usp=sharing) and the [MarsPlanBench](https://drive.google.com/drive/folders/1ID2xosJz7Cp0xIcWzMRZcCsb5OMhLtyf?usp=sharing) datasets, and place them inside the Datasets folder.


# MarsPlanBench

Runnin on selected maps (used for paper experiments):

	python run.py   --data-dir Datasets/MarsPlanBench/MarsPlanBench-10


Running on the full datasets:
	
	python run.py   --data-dir Datasets/MarsPlanBench/MarsPlanBench-10_raw
	
	python run.py   --data-dir Datasets/MarsPlanBench/MarsPlanBench-20_raw


Test run with background images:

	python run.py   --data-dir Datasets/MarsPlanBench/Background/npy   --background-png-dir Datasets/MarsPlanBench/Background/png


# MoonPlanBench

Full dataset used in paper experiments:

	python run.py   --data-dir Datasets/MoonPlanBench/MoonPlanBench-10
	
	python run.py   --data-dir Datasets/MoonPlanBench/MoonPlanBench-15
	
	python run.py   --data-dir Datasets/MoonPlanBench/MoonPlanBench-20


Test run with background images:

	python run.py   --data-dir Datasets/MoonPlanBench/Background/npy   --background-png-dir Datasets/MoonPlanBench/Background/jpeg --moon
