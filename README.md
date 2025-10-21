# 🚨 Cyber Threat Detection Web App (ANN-based)

This project implements a web application using **Streamlit** to demonstrate and compare different machine learning and Artificial Neural Network (ANN) algorithms for cyber threat detection, specifically using the **KDD Cup '99** dataset for network intrusion detection.

The application allows users to upload the dataset, run preprocessing steps like **TF-IDF**, and then train and evaluate a variety of classification models, including a custom ANN.

## 🌟 Features

* **Data Upload:** Allows users to upload the `kdd_train.csv` (or any compatible network intrusion dataset).
* **Preprocessing:** Includes functionality to run a **TF-IDF** algorithm for vector generation.
* **Model Comparison:** Provides options to run and compare performance metrics (Accuracy, Precision, Recall, F1-Score) of multiple models:
    * **Artificial Neural Network (ANN)**
    * **K-Nearest Neighbors (KNN)**
    * **Support Vector Machine (SVM)**
    * **Decision Tree (DT)**
    * **Random Forest (RF)**
    * **Naive Bayes (NB)**
* **Visualization:** Displays interactive bar graphs to compare model performance metrics visually.

## 🛠️ Project Structure

| File | Description |
| :--- | :--- |
| `app.py` | The main Streamlit web application script. Defines the layout, handles file uploads, button actions, and calls functions from `model_utils.py`. |
| `model_utils.py` | Contains all core machine learning logic: data preprocessing, model definitions (ANN, KNN, SVM, etc.), training, and evaluation functions. |
| `kdd_train.csv` | Sample dataset (KDD Cup '99) used for training and testing the threat detection models. |
| `requirements.txt` | Lists all necessary Python dependencies required to run the application. |

## 🚀 Getting Started

Follow these steps to set up and run the web application locally.

### Prerequisites

You need **Python (3.7+)** installed on your system.

### Installation

1.  **Clone the Repository (or download the files):**
    ```bash
    git clone [https://github.com/YourUsername/cyberthreat-detection-webapp.git](https://github.com/YourUsername/cyberthreat-detection-webapp.git)
    cd cyberthreat-detection-webapp
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # Windows: venv\Scripts\activate
    # Linux/macOS: source venv/bin/activate
    ```

3.  **Install Dependencies:**
    All required libraries (Streamlit, pandas, tensorflow, scikit-learn, etc.) are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

### Running the App

1.  Ensure you are in the project directory (`cyberthreat-detection-webapp`).
2.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

This command will automatically open the web application in your default browser (usually at `http://localhost:8501`).

## 💡 How to Use the App

1.  **Upload Data:** On the main page, upload the `kdd_train.csv` file using the file uploader.
2.  **Preprocess:** Click the **"Run Preprocessing TF-IDF Algorithm"** button to transform the categorical and text-based features into numerical vectors suitable for the models.
3.  **Generate Event Profile:** Click **"Generate Event Profiles & Split Data"** to perform final feature engineering and split the data into training and testing sets.
4.  **Run Models:** Use the buttons for each model (e.g., **"Run ANN Algorithm"**) to train the respective model and display its performance metrics.
5.  **Compare:** Click the **"Accuracy Comparison Graph"** or other comparison buttons to view a visual summary of the models you have run.

---

Feel free to customize the "Getting Started" section with your specific GitHub username and repository name once you have finished the upload!
