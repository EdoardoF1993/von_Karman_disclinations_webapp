import os
import yaml
import subprocess
from flask import Flask, render_template, request, jsonify, send_from_directory

# Inizializzazione Flask
#app = Flask(__name__)
app = Flask(__name__, static_folder="static", template_folder="templates")


DATA_FILE = 'Data.yml'
# Definisce la root directory dell'app Flask (la cartella 'src')
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Nuove costanti basate sulla struttura ad albero ---
# Parte del percorso del plot che viene gestita dall'endpoint Flask
#PLOT_FILENAME_RELATIVE = "transverseDisplacement.html"
PLOT_FILES = [
    "Airy.html",
    "transverseDisplacement.html",
    "visualization_sigma_rr.html"
]

# La directory che contiene i file di plot, relativa alla cartella 'src'
PLOT_DIRECTORY_RELATIVE_TO_SRC = os.path.join('..', 'output')


# --- Parametri di default (invariati) ---
DEFAULT_PARAMS = {
    'young_modulus_e': 100000.0,
    'poisson_ratio_nu': 0.15,
    'thickness_ratio_t': 0.1
}

def read_data():
    """Helper to safely read the YAML file, applying defaults if keys are missing."""
    data = {'frank_angle': [], 'positions_list': []}
    
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            file_data = yaml.safe_load(f)
            if file_data:
                data.update(file_data)
    
    for key, default_val in DEFAULT_PARAMS.items():
        if key not in data:
            data[key] = default_val

    if 'frank_angle' not in data: data['frank_angle'] = []
    if 'positions_list' not in data: data['positions_list'] = []

    return data

def write_data(data):
    """Helper to safely write to the YAML file."""
    with open(DATA_FILE, 'w') as f:
        yaml.dump(data, f, default_flow_style=None)

# --- Endpoint per servire i file di plot ---
# Questo endpoint permette al browser di accedere al file HTML di plot 
# anche se non si trova nella cartella 'static'.
@app.route('/plot_html/<path:filename>')
def serve_plot_html(filename):
    """Serve i file HTML del plot dalla directory di output."""
    plot_root_dir = os.path.join(PROJECT_ROOT, PLOT_DIRECTORY_RELATIVE_TO_SRC)
    
    # send_from_directory gestisce la ricerca del file all'interno della directory specificata
    return send_from_directory(plot_root_dir, filename)

@app.route('/')
def index():
    """Renderizza la pagina principale, passando l'URL del plot."""
    # Costruisce l'URL completo che punta all'endpoint sopra (es: /plot_html/path/to/file.html)
    plot_urls = {
        'plot1_url': f"/plot_html/{PLOT_FILES[0]}",
        'plot2_url': f"/plot_html/{PLOT_FILES[1]}",
        'plot3_url': f"/plot_html/{PLOT_FILES[2]}"
    }
    
    return render_template('index.html', **plot_urls)

@app.route('/get_data', methods=['GET'])
def get_data():
    data = read_data()
    return jsonify(data)

@app.route('/add_point', methods=['POST'])
def add_point():
    # ... (Logica invariata) ...
    try:
        content = request.json
        x = float(content['x'])
        y = float(content['y'])
        angle = float(content['angle'])
        
        e = float(content['E'])
        nu = float(content['nu'])
        t = float(content['t'])

        data = read_data()
        
        data['frank_angle'].append(angle)
        data['positions_list'].append([x, y])

        data['young_modulus_e'] = e
        data['poisson_ratio_nu'] = nu
        data['thickness_ratio_t'] = t

        write_data(data)
        return jsonify({"status": "success", "message": "Point added and parameters saved."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/set_params', methods=['POST'])
def set_params():
    # ... (Logica invariata) ...
    try:
        content = request.json
        data = read_data()
        
        data['model']['E'] = float(content['E'])
        data['model']['nu'] = float(content['nu'])
        data['model']['thickness'] = float(content['t'])

        write_data(data)
        return jsonify({"status": "success", "message": "Parameters updated."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    # ... (Logica invariata) ...
    try:
        reset_data = read_data()
        reset_data['frank_angle'] = []
        reset_data['positions_list'] = []
        
        write_data(reset_data)
        return jsonify({"status": "success", "message": "Points reset. Parameters retained."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/compute', methods=['POST'])
def compute():
    # ... (Logica invariata) ...
    try:
        cmd = ["python", "Demo.py"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        return jsonify({
            "status": "success", 
            "output": result.stdout
        })
    except subprocess.CalledProcessError as e:
        return jsonify({
            "status": "error", 
            "message": "Simulation failed.", 
            "details": e.stderr
        }), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists(DATA_FILE):
        write_data({'frank_angle': [], 'positions_list': [], **DEFAULT_PARAMS})
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
