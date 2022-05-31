# General Description

This project developed for ["Lifelong Self-Adaptation: Self-Adaptation Meets Lifelong Machine Learning"](https://arxiv.org/abs/2204.01834) paper, published at SEAMS@ICSE 2022. In this paper, the proposed archtiecture was validated 
on two varient domains: Internet-of-Things and Gas Delivery System. You can find these two validation cases in two different folders in `src`.
Moreover, extra materials (required data for IoT case) can be found [on the webiste of the paper](https://people.cs.kuleuven.be/~danny.weyns/software/LLSAS/).

# IoT Validation Case 

## Porject Structure

In the root folder, there are multiple folders:
  1.  `deltaiot_simulator` is the DeltaIoT simulator, the IoT system for validating the lifelong learning loop. 
  2.  `self_adaptation` implements MAPE-K functions (`feedback_loop.py`) with support of machine learning (machine learning component is implemented in `machine_learning.py`).
  3.  `lifelong_self_adaptation` implements a meta layer over the `self-adaptation` for the matter of lifelong self-adaptation. It comprisies:
      1.   knowledge component (`knowledge.py`),
      2.   task manager component (`task_manager.py`),
      3.   knowledge-based reasoner (a function in `lifelong_learning_loop.py`)
      4.   task-specific miner (a function in `lifelong_learning_loop.py`)
      5.   general lifelong learning loop (in `lifelong_learning_loop.py`)
  4.  `common` implements base classes that are shared between the adaptation layers. For example, goal types (`goal.py`) and learner types (`learning.py`). The other important part of this module is `stream` that mimics collecting data in each adaptation cycle.
  5.   `visualizer` implements some methods to visualize the adaptation result

## Requirements

This project is developed based on the key following tools and platforms:
  1. `Ubuntu 20.04 LTS`
  2. `Python 3.7`
     1. `Plotly` library for data visualization  
     2. `Orca` enging to render PDF files (https://community.plotly.com/t/using-orca-to-export-plotly-plots-to-pdf-files/18816)

Also, we used many other standard libraries in python like `numpy`, but there should not be a complexity for getting final resutls.  

## Run

To start the project, **first**, you need to run the DeltaIoT system simulator (in `deltaiot_simulator` folder) based on 
   - some desired configs in `deltaiot_simulator > SMCConfig.properties` file. (The current config shows `DeltaIoTv1` simulator with 1505 cycles.)
   - Another required modification before executing the simulator is changing the global interference and their affect over links in `deltaiot_simulator > activforms > src > mapek > FeedbackLoo.java` file. You can switch between no drift and under drift scenarios by setting the variable `is_drift_scenario` in the function `start()` of the same folder. 
   - To collect all quality services for all adaptation option, Uppaal-SMC has been utilized. To run this verifer, it is rquired to address `verifyta` file which provides the way of running the verifier through the command line. This file has been already located at `deltaiot_simulator > uppaal-verifyta > verifyta`. But, depending on the platofrms and their correspnding settings, you may need to refer into the file by an absolute addressing. To do that, you can place the absolute address of this file in `deltaiot_simulator > activforms > src > smc > SMCChecker` and at the starting point of `"./uppaal-verifyta", "verifyta -a %f -E %f -u %s"`.
   - Eventually, for running the simulator, you need to run the `Main.java` class in `deltaiot_simulator > activforms > src > main` folder. (If you install the VS code IDE and the `Extension Pack for Java` for it, by opening the project from the root of `deltaiot_simulator`, you can simply right click over the file name and click on `Run Java` to run the project.) You can also use `mvn exec:java -pl activforms` command to run the simulator by maven. More details of running the simulator can be found here `https://people.cs.kuleuven.be/~danny.weyns/software/machine-learning.htm`. 
You can have access to pre-generated data for both above-mentioned cases next to this project zip file on the same website. (It is also a zip file contains two folders for the two different scenarios.)   

**Second**, after running the simulator, some files will be generated in `deltaiot_simulator > output > deltaiotv1` (if the setup of the simulator in `SMCConfig.properties` is set on `DeltaIoTv1`). These files contains true (or simulated) values of qualities of the system for each adaptation option and value of uncertainties (links' interference and motes' load of messages) in each adaptation cycle. We should move them into a subfolder of the `./data` folder to be used for further feedback loop simulations.
Also, regarding to the paper, we need to run the first step twice, one for a no drift scenario and other for under derfit scenario. Hence, we should move the generated files each time, as they can be rewritten by the next running simulation. 
   

**Thrid**, after moving required data, we should the address of files in `fetch_stream()` function in `./main.py`. There are two scenarios inside this function that can be adjusted by the input parameter `with_drift` to it. After setting required addresses in this function, we are ready to  run a self-adaptation loop with/without meta-layer that is for lifelong self-adaptation. For this purpose, there are two functions in the `./main.py` called `run_feedbackloo` and `run_ll_loop`, respectively. Before calling these functions in the `./main.py`, you should set the address for their output files which will contain their predictions for quality of services for each adaptation option in each adaptation cycle. 
For running the desired loop, you only need to uncomment the corresponding function to the loop. For example, for running MAPE-K feedback loop without meta-layer lifelong learning, you need to call `run_feedbackloop()` function in the scope of `if __name__ == "__main__":`. Note that you need to adjust the address of the cycle data in the corresponding `Stream` object of the adaptation function. As another example, you can run the self-adaptation loo with lifelong self-adaptation meta-layer using `run_ll_loop()` function. Note that, you can also config more settings for the loops (MAPE-K and lifelong learning) via input parameters of their corresponding object. For example, for a lifelong learning loop, you can determine that the loop must evovle the MAPE-K loop by selecting the best function among one or two types of learner, using `bilearning_mode = False` and  `bilearning_mode = True`, respectively. 
Running of the  `./main.py` can be done by the following command based on the the root folder:

```sh
$ python3 main.py
```

**Eventually**, you can compare and visualize the output of loops' using `qos_visualization_multiple_goals_with_utility()` function in `main.py`. Note that the corresponding file addresses to prediction outputsfor different scenarios should be adjusted beforehand (in appropriate placeholders determined by `<path to the generated file>`).


These are main steps for simulating and comparing results of different scenarios under different type of loops (i.e., MAPE-K loop or the loop under lifelong learning with different settings). Also, there other functions to visulize training time in these scenarios and also representing global network interference (affects on SNRs) and task identification in different adaptation cycles by `snr_vis()`, `plot_learning_time()`, and `plot_drifts()` functions, respectively. Note that you should update address of the related generated files for each of these functions based on the exiting placeholders in those functions. 

## Results

For details of plots in the paper, you can look at the results in the `figures` folder. You can find more interactive figures in `.htm` formats in compare with `.pdf` versions. Also, name of these files are started with the `Fig#i` which `#i` indicates the figure number of plots in the paper or its relation to `#i`-th figure number. For example, `Fig5-energy_consumption` has not been included in the paper, but it is showing the same meaning of the figure 5 for the packet loss in the paper.     


# Gas Delivery System Validation Case 

## Porject Structure

In the root folder, there are multiple folders:
  1.  `self_adaptation` implements MAPE-K functions (`feedback_loop.py`) with support of machine learning (a classifier to predict class of sensory data).
  2.  `lifelong_self_adaptation` implements a meta layer over the `self-adaptation` for the matter of lifelong self-adaptation. It comprisies:
      1.   knowledge component (`knowledge.py`),
      2.   task manager component (`task_manager.py`),
      3.   knowledge-based reasoner (a function in `lifelong_learning_loop.py`)
      4.   task-specific miner (a function in `lifelong_learning_loop.py`)
      5.   general lifelong learning loop (in `lifelong_learning_loop.py`)
  3.  `common` implements base classes that are shared between the adaptation layers. The important part of this module is `stream` that mimics collecting data in each adaptation cycle (data from this link: http://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset+at+Different+Concentrations)
  4.   `visualizer` implements some methods to visualize the adaptation results in the paper.

## Requirements

This project is developed based on the key following tools and platforms:
  1. `Ubuntu 20.04 LTS`
  2. `Python 3.7`
     1. `Plotly` library for data visualization  
     2. `Orca` enging to render PDF files (https://community.plotly.com/t/using-orca-to-export-plotly-plots-to-pdf-files/18816)

Also, we used many other standard libraries in python like `numpy` and `keras_tuner`, but there should not be a complexity for getting final resutls.  

## Run

To start the project, you onlu need to run the following command from the root folder `python3 main.py`.

## Results

For details of plots in the paper, you can look at the results in the `figures` folder. You can find more interactive figures in `.htm` formats in compare with `.pdf` versions. Also, name of these files are started with the `Fig#i` which `#i` indicates the figure number of plots in the paper or its relation to `#i`-th figure number. For example, `Fig13_accuracy_over_time.pdf` has not been included in the paper, but it is showing the accuracy of different methods over time which their values are represented in different ranges in Fig13.

# Citation

<pre>
@inproceedings{gheibi2022lifelong,
    title={Lifelong Self-Adaptation: Self-Adaptation Meets Lifelong Machine Learning},
    booktitle={2022 International Symposium on Software Engineering for Adaptive and Self-Managing Systems (SEAMS)},
    year={2022},
    organization={ACM}
}
</pre>
