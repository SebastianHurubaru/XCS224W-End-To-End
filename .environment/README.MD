## Creating the Python environment

We provide below instructions to setup the environment used throughout this class. We are supporting Linux, macOS & Windows so please make sure you follow the exact instructions provided for your platform, in order to have it working on your machine!

<br />

### !! **The commands below need to be executed in a `bash` terminal on Linux/macOS and in a `cmd` terminal on Windows** !!

<br />


### Step 1: Make sure you have Miniconda installed. More info about Miniconda installation for your specific platform can be found [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install).

<br />

### Step 2: create a default conda environment named **XCS224W_E2E** with Python 3.10: 

```
conda create --name XCS224W_E2E python=3.10 -y
```

<br />

 
### Step 3: activate the just created XCS224W_E2E environment:

```
conda activate XCS224W_E2E
```

### Step 4: we are supporting the following devices on each of the platforms:

|             | `cpu` | `cu118` |
|-------------|-------|---------|
| **Linux**   | ✅    | ✅      |
| **Windows** | ✅    | ✅      |
| **macOS**   | ✅    |         |

<br />

If you have a Nvidia GPU the device you would need to set would be `cu118`, otherwise it should be `cpu`, as defined in the command below. 

**Apple GPU (`mps`) is not supported yet by Pytorch Geometrics(PyG)**!

<br />

**For Linux/MacOS**:

```
      
export DEVICE=cpu

```

<br />

**For Windows**:

```
      
set DEVICE=cpu

```

<br />

### Step 5: install first set of required tools in the **XCS224W_E2E** environment as defined in the `.environment/requirements.txt` file.

**For Linux/MacOS**:

```

pip install \
    -f https://download.pytorch.org/whl/torch_stable.html \
    -r .environment/requirements.txt

```

**For Windows**:

```

pip install ^
    -f https://download.pytorch.org/whl/torch_stable.html ^
    -r .environment\requirements.txt

```

<br />

### Step 6: install the second set of required tools in the **XCS224W_E2E** environment as defined in the `.environment/graph_ml_requirements.txt` file.

<br />

**For Linux/MacOS**:

```
      
pip install \
    -f https://data.pyg.org/whl/torch-2.0.1+${DEVICE}.html \
    -f https://pytorch-geometric.com/whl/torch-2.0.1+${DEVICE}.html \
    -r .environment/graph_ml_requirements.txt

```

<br />

**For Windows**:

```
      
pip install ^
    -f https://data.pyg.org/whl/torch-2.0.1+%DEVICE%.html ^
    -f https://pytorch-geometric.com/whl/torch-2.0.1+%DEVICE%.html ^
    -r .environment\graph_ml_requirements.txt

```
