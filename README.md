# AI-Enhanced MariaDB Jupyter Magic Commands with Vector RAG Integration🧠

### **Theme:** Innovation  

## 📘 About the Project  
This project transforms **MariaDB** into an intelligent AI/ML platform by introducing **Jupyter Magic Commands** that seamlessly integrate vector operations, Retrieval-Augmented Generation (RAG) pipelines, and machine learning models directly with database workflows. Built entirely in **Google Colab**, it bridges the gap between traditional SQL analytics and modern AI inference, empowering developers and data scientists to perform complex AI tasks using intuitive, SQL-like commands within a familiar Jupyter environment.  

## 🚀 Core Features  
- **`%%mariadb` Magic Command:** Executes SQL queries directly from Jupyter and returns results as Pandas DataFrames.  
- **Automatic Visualisation:** Generates charts using Matplotlib or Seaborn for numeric query results.  
- **`%%mariadb_vector` Extension:** Integrates Hugging Face embeddings with MariaDB Vector columns for semantic and similarity search.  
- **RAG (Retrieval-Augmented Generation) Pipeline:** Enables context-aware AI responses using database-driven document retrieval.  
- **Google Colab Support:** Complete setup and execution directly in Colab, requiring no local installation.  
- **Reusable Module:** Packaged code for quick integration in any AI or data workflow.  

## 🧩 Dataset Information  

This project uses the **[OpenFlights dataset](https://github.com/MariaDB/openflights)** as the primary data source.  
You can load it in Colab with:  

```bash
!wget https://raw.githubusercontent.com/MariaDB/openflights/main/airports.dat -O airports.csv
!wget https://raw.githubusercontent.com/MariaDB/openflights/main/routes.dat -O routes.csv
````

These datasets include information about airports, airlines, and global routes — perfect for demonstrating data querying, visualization, and vector-based similarity search (e.g., *“Find airports similar to Delhi”*).

## ⚙️ Setup Instructions (Google Colab)

1. **Install MariaDB and Dependencies**

   ```bash
   !apt-get update
   !apt-get install -y mariadb-server
   !pip install mariadb sqlalchemy pandas matplotlib seaborn transformers
   !service mysql start
   ```

2. **Create and Load Database**

   ```python
   import pandas as pd, mariadb
   conn = mariadb.connect(user='root', password='', database='test_db')
   cursor = conn.cursor()
   cursor.execute("CREATE DATABASE IF NOT EXISTS test_db;")
   ```

3. **Run SQL Queries via Magic Command**

   ```python
   %%mariadb
   SELECT Country, COUNT(*) AS AirportCount
   FROM airports
   GROUP BY Country
   ORDER BY AirportCount DESC
   LIMIT 10;
   ```

4. **Visualize Results**

   ```python
   import matplotlib.pyplot as plt
   df.plot(kind='bar', x='Country', y='AirportCount')
   plt.title('Top 10 Countries by Number of Airports')
   plt.show()
   ```

## Data Visualisation Insights

<img width="790" height="1989" alt="maria13" src="https://github.com/user-attachments/assets/8cd91529-efe3-4447-8b0d-23ce4cec364c" />
<img width="790" height="1989" alt="maria12" src="https://github.com/user-attachments/assets/ecf10f89-63c5-4f2f-afea-445871979330" />
<img width="1077" height="314" alt="mariaaa" src="https://github.com/user-attachments/assets/7443c14e-1016-424a-8191-b01f21d1a828" />

<img width="650" height="671" alt="Maria1" src="https://github.com/user-attachments/assets/5b116f56-8925-4254-b9bb-8cc8a87fb625" />
<img width="650" height="671" alt="Maria2" src="https://github.com/user-attachments/assets/cbff737c-2c31-497c-8ec3-87c70065cca7" />
<img width="650" height="671" alt="maria3" src="https://github.com/user-attachments/assets/e4c837d8-9341-438d-82dd-f34593bb5811" />

<img width="554" height="427" alt="maria8" src="https://github.com/user-attachments/assets/df1df448-cf13-4268-9a34-7d12aa839b39" />
<img width="554" height="427" alt="maria9" src="https://github.com/user-attachments/assets/71b87ca0-f2c5-45cd-832b-fb77df0eecf6" />
<img width="554" height="427" alt="maria10" src="https://github.com/user-attachments/assets/68dbd71c-23ba-42b7-ba4c-0cb6c69afdc7" />
<img width="561" height="427" alt="maria11" src="https://github.com/user-attachments/assets/dbc91729-410a-4b9f-a4ed-a0b269bf6faa" />


## 🧠 AI / Vector Integration (Optional Advanced Feature)

Use `%%mariadb_vector` to generate and store embeddings from text fields (e.g., airport names or cities) using the **Hugging Face** model `sentence-transformers/all-MiniLM-L6-v2`, enabling semantic similarity and RAG-based querying.

## 📊 Additional Sample Datasets

This directory includes a few sample datasets to get you started:

* **`california_housing_data*.csv`** – California housing data from the 1990 US Census
  More info: [Census Dataset Documentation](https://docs.google.com/document/d/e/2PACX-1vRhYtsvc5eOR2FWNCwaBiKL6suIOrxJig8LcSBbmCbyYsayia_DvPOOBlXZ4CAlQ5nlDD8kTaIDRwrN/pub)

* **`mnist_*.csv`** – A small sample of the MNIST handwritten digits dataset
  Description: [Yann LeCun’s MNIST Page](http://yann.lecun.com/exdb/mnist/)

* **`anscombe.json`** – Contains a copy of *Anscombe’s Quartet*
  Source: Anscombe, F. J. (1973). *Graphs in Statistical Analysis*. *American Statistician*, 27(1): 17–21. [JSTOR 2682899](https://www.jstor.org/stable/2682899).
  Prepared via the `vega_datasets` library.

## 🧩 Demo Notebook

A ready-to-use Colab demo notebook includes:

* Environment setup
* Dataset loading
* Example SQL queries and plots
* AI/Vector and RAG integration demos

➡️ **Open in Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/mariadb-jupyter-magic/blob/main/demo.ipynb)

## 📚 License

This project is open-source and released under the **MIT License**.
