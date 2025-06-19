import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, request, render_template_string, send_from_directory
from pca import compress_image_pca
import numpy as np
import cv2
from PIL import Image
import io, time, base64

app = Flask(__name__)

# Serve aboutus images
ABOUTUS_FOLDER = os.path.join(app.root_path, 'aboutus')
@app.route('/aboutus/<path:filename>')
def aboutus_files(filename):
    return send_from_directory(ABOUTUS_FOLDER, filename)

# Full UI template with correct image paths
template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>PCA Image Compression Demo</title>
    <script src="https://cdn.jsdelivr.net/npm/particles.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html, body { height: 100%; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        #particles-js { position: fixed; width: 100%; height: 100%; z-index: -1; }
        body { background: #222; color: #333; }
        .container { max-width: 1000px; margin: 3rem auto; background: rgba(255,255,255,0.95); padding: 2rem; border-radius: 10px; box-shadow: 0 8px 24px rgba(0,0,0,0.2); }
        h1 { text-align: center; margin-bottom: 1.5rem; }
        form { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        .form-group { display: flex; flex-direction: column; }
        label { margin-bottom: .5rem; font-weight: 600; }
        input[type=file] { padding: .5rem; border: 2px dashed #aaa; border-radius: 6px; background: #fafafa; }
        .slider-container { grid-column: span 2; }
        input[type=range] { -webkit-appearance: none; width: 100%; height: 8px; border-radius: 5px; background: #ddd; outline: none; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 20px; height: 20px; border-radius: 50%; background: #4caf50; cursor: pointer; transition: background 0.3s ease; }
        input[type=range]:hover::-webkit-slider-thumb { background: #45a049; }
        .percentage { text-align: right; font-weight: bold; margin-top: .25rem; }
        .scale-bar { display: flex; justify-content: space-between; margin-top: .5rem; }
        .scale-bar span { flex:1; text-align: center; padding:.3rem; border-radius:4px; color:#fff; opacity:.5; transition:opacity .2s ease; }
        .scale-low { background:#4caf50; }
        .scale-medium { background:#ffeb3b; color:#333; }
        .scale-high { background:#ff9800; }
        .scale-veryhigh { background:#f44336; }
        button { grid-column: span 2; padding:.75rem; background:#4caf50; color:#fff; border:none; border-radius:6px; cursor:pointer; transition:background .2s ease; }
        button:hover { background:#388e3c; }
        .results { margin-top:2rem; }
        .stats { display:flex; gap:1rem; margin-bottom:1.5rem; }
        .stats div { flex:1; background:#fafafa; padding:1rem; border-radius:6px; text-align:center; }
        .images { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
        .img-box { background:#fff; border-radius:6px; overflow:hidden; }
        .img-box img { width:100%; cursor:zoom-in; transition:transform .3s ease; }
        .img-box img:active { transform:scale(2); cursor:zoom-out; }
        .img-box p { padding:.5rem; text-align:center; color:#555; }
        .download-btn { display: block; width: 60%; margin: 1rem auto; padding: .75rem 1.5rem; background: #2196f3; color: #fff; border-radius: 6px; text-decoration: none; text-align: center; transition: background 0.2s ease; }
        .download-btn:hover { background:#1976d2; }
        .charts { display:flex; justify-content:center; gap:2rem; margin-top:2rem; }
        .charts img { max-width:45%; border-radius:6px; box-shadow:0 4px 12px rgba(0,0,0,0.1); }
        .about { margin-top:3rem; padding:1.5rem; background:#e8f4fd; border-radius:6px; text-align:center; }
        .profiles { display:flex; justify-content:center; gap:2rem; margin-top:1rem; }
        .profile { width:150px; }
        .profile img { width:auto; height:150px; object-fit:cover; max-width: 100%; border-radius:50%; border:4px solid #fff; }
        .profile h4 { margin-top:.75rem; }
        .profile p { font-size:.9rem; color:#555; }
    </style>
    <script>
        particlesJS.load('particles-js', 'https://cdn.jsdelivr.net/gh/VincentGarreau/particles.js/particles.json');
        function updateScale(val) {
            document.getElementById('lbl-low').style.opacity = val <= 25 ? '1' : '0.5';
            document.getElementById('lbl-med').style.opacity = val > 25 && val <= 50 ? '1' : '0.5';
            document.getElementById('lbl-high').style.opacity = val > 50 && val <= 75 ? '1' : '0.5';
            document.getElementById('lbl-vh').style.opacity = val > 75 ? '1' : '0.5';
            document.getElementById('perc').innerText = val;
        }
    </script>
</head>
<body>
    <div id="particles-js"></div>
    <div class="container">
        <h1>PCA Image Compression</h1>
        <form method="POST" enctype="multipart/form-data">
            <div class="form-group">
                <label for="image_file">Choose Image</label>
                <input id="image_file" type="file" name="image_file" required>
            </div>
            <div class="slider-container">
                <label for="components">Components: <span id="perc">{{ quality }}</span>%</label>
                <input id="components" type="range" name="quality" min="0" max="100" value="{{ quality }}" oninput="updateScale(this.value)" onchange="updateScale(this.value)">
                <div class="scale-bar">
                    <span id="lbl-low" class="scale-low">Low</span>
                    <span id="lbl-med" class="scale-medium">Medium</span>
                    <span id="lbl-high" class="scale-high">High</span>
                    <span id="lbl-vh" class="scale-veryhigh">Very High</span>
                </div>
            </div>
            <button type="submit">Compress Now</button>
        </form>
        {% if original and compressed %}
        <div class="results">
            <div class="stats">
                <div><strong>Runtime:</strong> {{ runtime_ms }} ms</div>
                <div><strong>Original:</strong> {{ size_orig }} B</div>
                <div><strong>Compressed:</strong> {{ size_comp }} B</div>
            </div>
            <div class="images">
                <div class="img-box">
                    <h4>Original</h4>
                    <img src="data:image/png;base64,{{ original }}">
                    <p>{{ size_orig }} bytes</p>
                </div>
                <div class="img-box">
                    <h4>Compressed</h4>
                    <img src="data:image/png;base64,{{ compressed }}">
                    <p>{{ size_comp }} bytes</p>
                    <a download="compressed.png" href="data:image/png;base64,{{ compressed }}" class="download-btn">Download Compressed</a>
                </div>
            </div>
            <div class="charts">
                <img src="data:image/png;base64,{{ bar_chart }}" alt="Bar Chart">
                <img src="data:image/png;base64,{{ pie_chart }}" alt="Pie Chart">
            </div>
        </div>
        {% endif %}
        <div class="about">
            <h3>About Us</h3>
            <div class="profiles">
                <div class="profile">
                    <img src="/aboutus/faris.jpg" alt="Faris">
                    <h4>Mufti Faris Murtadho</h4>
                    <p>NIM: L0124133</p>
                    <p>Role: PCA & Backend</p>
                </div>
                <div class="profile">
                    <img src="/aboutus/yashif.jpeg" alt="Yashif">
                    <h4>Yashif Victoriawan</h4>
                    <p>NIM: L0124124</p>
                    <p>Role: Frontend & UI/UX</p>
                </div>
                <div class="profile">
                    <img src="/aboutus/yusran.jpeg" alt="Yusran">
                    <h4>Yusran Rizqi Laksono</h4>
                    <p>NIM: L0124125</p>
                    <p>Role: Docs & Testing</p>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET','POST'])
def index():
    quality = 50
    original = compressed = bar_chart = pie_chart = None
    size_orig = size_comp = runtime_ms = 0
    if request.method=='POST':
        quality = int(request.form.get('quality', 50))
        data = np.frombuffer(request.files['image_file'].read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        # Original
        buf_o = io.BytesIO()
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(buf_o, format='PNG')
        data_o = buf_o.getvalue(); size_orig=len(data_o)
        original = base64.b64encode(data_o).decode()
        # PCA + JPEG
        img_pca = compress_image_pca(img, quality)
        start = time.time(); buf_c = io.BytesIO()
        Image.fromarray(cv2.cvtColor(img_pca, cv2.COLOR_BGR2RGB)).save(buf_c, format='JPEG', quality=quality)
        runtime_ms = int((time.time()-start)*1000)
        data_c = buf_c.getvalue(); size_comp=len(data_c)
        compressed = base64.b64encode(data_c).decode()
        # Bar chart
        plt.figure(); plt.bar(['Original','Compressed'], [size_orig,size_comp]); plt.ylabel('Bytes')
        tmp = io.BytesIO(); plt.tight_layout(); plt.savefig(tmp, format='png'); plt.close()
        bar_chart = base64.b64encode(tmp.getvalue()).decode()
        # Pie chart
        plt.figure(); plt.pie([size_comp, size_orig-size_comp], labels=['Compressed','Reduced'], autopct='%1.1f%%'); plt.title('Ratio')
        tmp2 = io.BytesIO(); plt.tight_layout(); plt.savefig(tmp2, format='png'); plt.close()
        pie_chart = base64.b64encode(tmp2.getvalue()).decode()
    return render_template_string(template,
        quality=quality,
        original=original,
        compressed=compressed,
        size_orig=size_orig,
        size_comp=size_comp,
        runtime_ms=runtime_ms,
        bar_chart=bar_chart,
        pie_chart=pie_chart)

if __name__=='__main__':
    app.run(debug=True, port=5000)
