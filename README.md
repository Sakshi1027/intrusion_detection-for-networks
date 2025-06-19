# Intrusion Detection for Networks using GNN and Causal Sampling

This repository contains a project implementing intrusion detection systems using **Graph Neural Networks (GNNs)** and **causal sampling** techniques. It includes specialized models for detecting **malware**, **IoT attacks**, **phishing**, **DDoS attacks**, and a **global model** integrating all attack types. The models are trained on network data represented as graphs, leveraging PyTorch Geometric for GNN implementation and causal sampling to address class imbalance and temporal dynamics. All datasets are derived from the CICIDS2017 dataset, but users must download and process them independently.

## Repository Structure

```
intrusion-detection-gnn/
├── notebooks/
│   ├── malware_detection.ipynb  # Malware detection model
│   ├── iot_detection.ipynb      # IoT attack detection model
│   ├── phishing_detection.ipynb # Phishing detection model
│   ├── ddos_detection.ipynb     # DDoS attack detection model
│   └── global_model.ipynb       # Global model for all attack types
├── models/
│   ├── malware_model.pth        # Trained malware model weights
│   ├── iot_model.pth            # Trained IoT model weights
│   ├── phishing_model.pth       # Trained phishing model weights
│   ├── ddos_model.pth           # Trained DDoS model weights
│   └── global_model.pth         # Trained global model weights
├── requirements.txt              # Python dependencies
└── README.md                    # Project overview (this file)
```

## Datasets

All models (malware, IoT, phishing, DDoS, global) use datasets derived from the CICIDS2017 dataset, provided by the Canadian Institute for Cybersecurity. Datasets are not included in this repository. Users must download CICIDS2017 and process it into Excel format (`.xlsx`) for compatibility with the notebooks.

### Dataset Details

- **Source**: CICIDS2017, containing labeled network traffic (PCAP and CSV files) for various intrusion types.
- **Usage Terms**: For research purposes only. Cite the dataset in publications:

  ```
  Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. In ICISSP (pp. 108-116).
  ```
- **Access**:
  1. Visit https://www.unb.ca/cic/datasets/ids-2017.html.
  2. Register and request access to download `MachineLearningCSV.zip` (\~2.8GB).
  3. Select relevant CSVs:
     - **Malware**: Use files like `Wednesday-WorkingHours.pcap_ISCX.csv`.
     - **IoT**: Filter IoT-related traffic (e.g., specific protocols or devices).
     - **Phishing**: Use web attack data (e.g., `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`).
     - **DDoS**: Use `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`.
     - **Global**: Combine multiple CSVs or use a representative subset.
- **Expected Format**:
  - **File Format**: Excel (`.xlsx`).
  - **Columns**:
    - `Filename`: Labels (e.g., `Spyware-TIBS`, `DDoS-Attack`). Extract prefix before hyphen (e.g., `Spyware`) for labels.
    - Other columns: Numerical features (e.g., packet counts, bytes, flow duration).
  - **Example**:
    - Columns: `Filename`, `Feature1`, `Feature2`, ..., `FeatureN`.
    - Sample row: `Spyware-TIBS`, `123.45`, `678.90`, ..., `0.12`.
  - **Size**: Processed `.xlsx` files are typically \~10MB, with enough rows for time windows (size=10, e.g., 851 rows for 85 windows).
- **Processing**:
  - Convert CSVs to `.xlsx` using pandas:

    ```python
    import pandas as pd
    df = pd.read_csv("path/to/cicids2017.csv")
    df["Filename"] = df["Label"].map(lambda x: f"{x}-ID")  # Customize mapping
    df = df[["Filename"] + [col for col in df.columns if col != "Label" and df[col].dtype in ["int64", "float64"]]]
    df.to_excel("processed_data.xlsx", index=False)
    ```
  - For OOD datasets, process a separate CSV (e.g., a different day’s data) similarly.
  - Ensure numerical features and sufficient rows for time windows.

## Setup

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/intrusion-detection-gnn.git
   cd intrusion-detection-gnn
   ```

2. **Install Dependencies**: Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

   **Note**: `requirements.txt` includes `torch==2.6.0+cu124` (CUDA 12.4). For CPU setups, use `torch==2.6.0`. Adjust `torch-scatter` and `torch-sparse` URLs via PyG’s wheel index.

3. **Google Colab (Recommended)**:

   - Open notebooks in Colab (see Usage).
   - Run installation cells to ensure compatibility.

4. **Prepare Datasets**:

   - Download CICIDS2017 and process into `.xlsx` files (see Datasets).
   - Store in Google Drive (e.g., `/content/drive/MyDrive/Malware/NEWOutput1.xlsx`).
   - Update notebook paths to match your file locations.

5. **Trained Models**:

   - Model weights (`.pth` files) are in `models/`.
   - If models exceed 100MB, use Git LFS or Google Drive (update paths in notebooks).
   - Example: `/content/drive/MyDrive/Malware/causal_graphsage_model.pth`.

## Usage

Run notebooks in Google Colab for training and evaluation:

- **Malware Detection**:

  ![Open in Colab](https://colab.research.google.com/drive/148WIPYjWcFKRYuoN4lRUPZU-4RHo6h_m)
- **IoT Attack Detection**:

  ![Open in Colab](https://colab.research.google.com/drive/1T-6EYaz9g2YHIo7cZZkYPPIcXR5nkBS-)
- **Global Model**:

  ![Open in Colab](https://colab.research.google.com/drive/1CjshuZKrci23AwhK64z-viUHs_YfgedP)

### Steps to Run a Notebook

1. Open the notebook in Colab using the badge above.
2. Mount Google Drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Install dependencies via notebook cells (e.g., `!pip install torch torch-geometric ...`).
4. Update dataset paths (e.g., `/content/drive/MyDrive/Malware/NEWOutput1.xlsx`).
5. Run cells to preprocess data, train, or evaluate (including OOD testing).

### Model Details

- **Architecture**: GraphSAGE (PyTorch Geometric) with causal sampling for temporal graph learning.
- **Data Processing**:
  - Encode labels with `LabelEncoder` (extract prefix from `Filename`).
  - Scale features with `StandardScaler`.
  - Create graphs with k=5 nearest neighbors (cosine similarity).
  - Split data into time windows (size=10).
- **Training**: 80% in-distribution training, 20% validation, OOD testing.
- **Evaluation**: Metrics include accuracy, precision, recall, F1-score, with MC Dropout for OOD uncertainty.
- **Causal Sampling**: Addresses class imbalance and temporal dependencies.

## Trained Models

- Models are saved as `.pth` files in `models/`.
- Example: `causal_graphsage_model.pth` for malware.
- Load models:

  ```python
  model.load_state_dict(torch.load('/path/to/model.pth'))
  ```
- For large files (&gt;100MB), use Git LFS or Google Drive.

## Results

- **Malware Detection** (from notebook):
  - Normal Inference: Accuracy: 0.5657, Precision: 1.0000, Recall: 0.5657, F1: 0.7226
  - MC Dropout Inference: Accuracy: 0.5114, Precision: 1.0000, Recall: 0.5114, F1: 0.6767
- Other models (IoT, phishing, DDoS, global) have similar evaluation pipelines. See notebooks for results.

## Permissions

This repository does not include a license. For permission to use, modify, or distribute code or models, contact sakshias.cs23@rvce.edu.in.

## Acknowledgments

- Built using PyTorch and PyTorch Geometric.
- Inspired by research on GNNs for network security and causal inference.

Replace `Sakshi1027` with your GitHub username in Colab badge URLs and clone command. Replace `sakshias.cs23@rvce.edu.in` with your contact email.
