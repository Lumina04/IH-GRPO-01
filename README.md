# IH-GRPO Algorithm Implementation

## Overview

This repository contains the implementation of the IH-GRPO algorithm.

## Environment Setup

To get started, set up the environment and install the necessary dependencies. These dependencies are listed in the `requirements.txt` file provided in this repository.

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/xxx/IH-GRPO.git
   cd IH-GRPO
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv ih_grpo
   source ih_grpo/bin/activate
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Algorithm

### Step 1: Start the API Server

First, start the API server by running the following command in one terminal window:

```bash
uvicorn sandbox_api:app --host 127.0.0.1 --port 12345 --workers 8
```

### Step 2: Modify the Hierarchical Loss Coefficient $\lambda$

You may modify the hierarchical loss coefficient $\lambda$ by updating it directly in the file `/verl/workers/actor/dp_actor.py` at line 469.

### Step 3: Run Training Script

In a separate terminal window, run the following script to start the training process:

```bash
bash ./example/delay_tir_exp/run_ih_train.sh
```

## Evaluation

### Step 1: Start the API Server

If you haven't already started the API server, run the following command in a terminal:

```bash
uvicorn sandbox_api:app --host 127.0.0.1 --port 12345 --workers 8
```

### Step 2: Run Evaluation Script

To evaluate the trained model, run the following command:

```bash
python3 ./examples/eval_hf.py
```