# <p align="center">🚀 Image Compression with PCA and Flask</p>

<!--
<p align="center">
  <a href="[URL_DEMO]" target="_blank">🔍 Live Demo Program</a> ·
</p>
-->

---

## 📖 Brief Description
A web application that demonstrates image compression using Principal Component Analysis (PCA) on face images. Users can upload an image, adjust the compression quality via a slider (0–100%), and see the original and compressed images side-by-side, complete with runtime statistics and visual charts.  

---

## 👥 Team Members

| Name                 | NIM / ID        |
| -------------------  | --------------- |
| Mufti Faris Murtadho | L0124133        |
| Yashif Victoriawan   | L0124124        |
| Yusran Rizqi Laksono | L0124125        |

---

## 🧰 Technologies Used

- **Language**: Python 3.9+
- **Framework**: Flask
- **Libraries & Frameworks**:  
  - OpenCV  
  - NumPy  
  - scikit-learn 
  - matplotlib 
  - Pillow  
- **Tools & Platforms**:  
  - Git & GitHub  
  - GitHub Actions (CI/CD)

---

## 📁 Structure Project

```text
IMAGE-COMPRESSION-MAIN/
├── requirements.txt
├── main.py
├── pca.py
│
├── __pycache__/
|   └── pca.cpython-312.pyc
|
├── aboutus/
│   ├── yashif.jpeg
│   ├── yusran.jpeg
|   └── faris.JPG
|
├── ss/
│   ├── mainmenu.png
│   ├── image.png
|   └── graph.png
|
└── .gigignore
```
---

## ⚙️ Setup Instructions

You can setup your project by cloning this repository and install the libraries above.

### 1. Clone the repository:
```bash
git clone <repository-url>
cd IMAGE-COMPRESSION-FLASK
```
### 2. (Optional) Create and activate a virtual environment:
<pre markdown><code>python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
</code></pre>  

### 3. Install dependencies:

```bash
pip install -r requirements.txt
```
### 4. Run the application:

```bash
python main.py
```

The app will be available at http://localhost:5000.

---

## Screenshots
### Figure 1
MAIN MENU

![Figure 1: Main Menu](ss/mainmenu.png)

### Figure 2
IMAGE

![Figure 2: Result](ss/image.png)

### Figure 3
GRAPH

![Figure 3: Result](ss/graph.png)

---

## NOTE
