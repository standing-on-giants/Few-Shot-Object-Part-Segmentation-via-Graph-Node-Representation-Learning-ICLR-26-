# Learning Query-Support Representations for Few-Shot Part Segmentation

This repository contains our framework for **Few-Shot Part Segmentation** using Graph Neural Networks (GNNs) with **MinCut Pooling**, leveraging dense features from **DINOv2** and superpixel graph representations.

---

## 🛠️ Docker Setup & Execution

Since the raw datasets and model checkpoints are large, we use **Volume Mounts** to give the Docker container direct access to the host's files. This keeps the Docker image lightweight and ensures that training progress/results are saved directly to your host machine.

### Step 1: Place Data & Pickle Files
Before running the container, ensure your repository root contains the required dataset folders and dictionary pickles:
```
part_ours_training/
├── data/
│   ├── training_data_MOHAN/          # ADE20K Training set
│   ├── testing_data_MOHAN/           # ADE20K Testing set
│   └── testing_data_pascal_MOHAN/    # PASCAL Testing set
├── new_dict.pkl                      # ADE20K Training Support Dictionary
├── new_dict_val.pkl                  # ADE20K Validation/Test Support Dictionary
└── pascal_val.pkl                    # PASCAL Test Support Dictionary
```

### Step 2: Build the Docker Image
Build the container from the root directory (large datasets, git history, and output folders will be ignored during build to keep it fast):
```bash
docker build -t few-shot-part-seg .
```

### Step 3: Run the Docker Container
Launch the container with GPU support enabled, mounting your host repository directory into the `/workspace` folder inside the container:
```bash
docker run --gpus all -it -v $(pwd):/workspace few-shot-part-seg
```

---

## 🚀 Running the Code inside the Container

Once inside the Docker container, you can run the primary training and testing scripts:

### Option A: Train the Model
Train the GNN model using cross-entropy and unsupervised MinCut Pooling losses:
```bash
python new/train1_new.py
```
*   **Checkpoints:** Saved dynamically on the host under `new/new_models/`.

### Option B: Run Inference & Evaluation
Evaluate the GNN model's predictions and compute the NMI and mIoU metrics:
```bash
python new/test_new.py
```
*   **Outputs:** Evaluation results (`.pt` files) are written directly to `new/new_models/OUTPUT_.../` on your host.
