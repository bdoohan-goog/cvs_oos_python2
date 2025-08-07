# Video Analysis and Inventory Management System

This project is a web-based application that analyzes videos of store shelves to identify products and determine their stock status. It uses Google Cloud services, including Vertex AI for video analysis, BigQuery for data storage, and Pub/Sub for messaging.

## Features

*   **Video Upload and Processing:** Users can select videos from a Google Cloud Storage bucket for analysis.
*   **Product Identification:** The application uses a Gemini model to identify products in the video and extract information such as product name, brand, and price.
*   **Out-of-Stock Detection:** The system can identify out-of-stock items and generate timestamps for each occurrence.
*   **Data Storage:** Analysis results are stored in a BigQuery table for further analysis and reporting.
*   **Web Interface:** A Flask-based web interface allows users to interact with the application, view results, and see related images.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

*   [Docker](https://www.docker.com/get-started)
*   [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
*   A Google Cloud project with the following APIs enabled:
    *   Vertex AI API
    *   BigQuery API
    *   Pub/Sub API
    *   Cloud Storage API
    *   Discovery Engine API

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://your-repository-url.git
    cd your-repository-directory
    ```

2.  **Configure Google Cloud authentication:**

    ```bash
    gcloud auth application-default login
    ```

3.  **Build the Docker image:**

    ```bash
    docker build -t video-analysis-app .
    ```

4.  **Run the Docker container:**

    ```bash
    docker run -p 8080:8080 -e GOOGLE_CLOUD_PROJECT=<YOUR_PROJECT_ID> video-analysis-app
    ```

    Replace `<YOUR_PROJECT_ID>` with your Google Cloud project ID.

## Usage

1.  Open your web browser and navigate to `http://localhost:8080`.
2.  Select a video from the dropdown list.
3.  Click the "Process Video" button to start the analysis.
4.  View the results in the "Gemini Output" and "BigQuery Preview" sections.
