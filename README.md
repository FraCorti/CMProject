

<img src="https://github.com/GiovanniSorice/MLProject/blob/master/logo/neuradillo.jpg" height="200" width="300">


# Neuradillo 
For the Computational Mathematics for Learning and Data Analysis [course](https://esami.unipi.it/esami2/programma.php?c=42267&aa=2019&docente=FRANGIONI&insegnamento=&sd=) we expanded the functionality of [Neuradillo](https://github.com/GiovanniSorice/MLProject). Using the backpropagation algorithm to compute the network gradient, we added [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) (based on this [paper](https://github.com/FraCorti/CMProject/blob/master/paper/quasi-newton-NN-CM.pdf)) and [Proximal Bundle](https://en.wikipedia.org/wiki/Subgradient_method#Subgradient-projection_&_bundle_methods) algorithms to train a neural network. 

More information about the theory and the convergence analysis regarding the algorithms can be found in the [report](https://github.com/GiovanniSorice/MLProject/blob/master/docs/report/relazione.pdf).
## Getting started

### Prerequisites 
The project use [Cmake 3.16](https://cmake.org/) as building system, [Conan](https://conan.io/) as package manager and [GCC](https://gcc.gnu.org/) as compiler. The project configuration can be changed by editing the [CMakeLists.txt](https://github.com/FraCorti/CMProject/blob/master/CMakeLists.txt) file, however the results we got may be different from yours due to the lack of optimization given by the -O3 [flag](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html) and matrix operation speedup given by [OpenMp](https://en.wikipedia.org/wiki/OpenMP).  

### Armadillo installation 
1. Clone the following repo: https://github.com/darcamo/conan-armadillo;
2. Inside the cloned repo run: `conan create . armadillo/stable`
3. If Armadillo is installed correctly an example program is execute and you can start use it [through Conan](https://docs.conan.io/en/latest/using_packages/conanfile_txt.html#requires).

### Gurobi installation 

## Usage


### Running the project
If CMake and Armadillo were installed correctly you have to create a directory where cmake store the configuration files needed to run the project:
`mkdir build && cd build `

Inside the build folder to generate the files that are needed by Conan type: 
 `conan install ..`

Then the for the CMake files give: 
 `cmake ..` 

In the end to build the project type: 
 `cmake --build .` 

If all the process is done correctly a */bin* folder is created with a binary file inside. This file can be executed with:
`./MLProject`

The error of the training set and validation set is print during the execution.

## Results
Here we show a learning curve plot we obtained during the training phase. 
 
<img src="/docs/report/img/Cup_loss_Reg_Zoom.png" height="50%" width="50%">

## Authors
* **Giovanni Sorice**  :computer: - [Giovanni Sorice](https://github.com/GiovanniSorice)
* **Francesco Corti** :nerd_face: :computer: - [FraCorti](https://github.com/FraCorti)
