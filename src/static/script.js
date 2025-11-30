// Costanti DOM
const canvas = document.getElementById('simCanvas');
const ctx = canvas.getContext('2d');
const statusDiv = document.getElementById('statusMsg');
const coordDiv = document.getElementById('coords');
const btnAdd = document.getElementById('btnAdd');
const outputWindow = document.getElementById('outputWindow');

// Slider
const sliderE = document.getElementById('sliderE');
const valueE = document.getElementById('valueE');
const sliderNu = document.getElementById('sliderNu');
const valueNu = document.getElementById('valueNu');
const sliderT = document.getElementById('sliderT');
const valueT = document.getElementById('valueT');

// Plot IFRAMES
const plotFrames = [
    document.getElementById('plot2'),
    document.getElementById('plot3'),
    document.getElementById('plot4')
];

// CONFIGURAZIONE CANVAS
const CANVAS_SIZE = 300;
const CENTER = CANVAS_SIZE / 2;
const RADIUS_PX = 140; 

let isAddMode = false;
let storedPoints = []; 

// --- FUNZIONI DI GESTIONE PARAMETRI E API ---

function getParams() {
    return {
        E: parseFloat(sliderE.value),
        nu: parseFloat(sliderNu.value),
        t: parseFloat(sliderT.value)
    };
}

function updateParam(paramKey) {
    const params = getParams();
    switch (paramKey) {
        case 'E':
            valueE.innerText = params.E.toFixed(0);
            break;
        case 'nu':
            valueNu.innerText = params.nu.toFixed(2);
            break;
        case 't':
            valueT.innerText = params.t.toFixed(2);
            break;
    }
}

async function saveParams() {
    const params = getParams();
    try {
        const response = await fetch('/set_params', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        const res = await response.json();
        if (res.status === 'success') {
            statusDiv.innerText = "Parameters saved.";
        } else {
            statusDiv.innerText = "Error saving parameters: " + res.message;
        }
    } catch (err) {
        console.error(err);
    }
}

window.loadInitialData = async function() {
    try {
        const response = await fetch('/get_data');
        const data = await response.json();

        // Load Parameters
        sliderE.value = data.young_modulus_e;
        sliderNu.value = data.poisson_ratio_nu;
        sliderT.value = data.thickness_ratio_t;
        updateParam('E');
        updateParam('nu');
        updateParam('t');

        // Load Points
        const R = RADIUS_PX;
        data.positions_list.forEach((pos, index) => {
            const xCart = pos[0];
            const yCart = pos[1];
            
            const xPx = xCart * R + CENTER;
            const yPx = -yCart * R + CENTER;
            
            storedPoints.push({ 
                xPx: xPx, 
                yPx: yPx, 
                angle: data.frank_angle[index] 
            });
        });
        renderPoints();
        statusDiv.innerText = `Loaded ${storedPoints.length} points and parameters.`;

    } catch (error) {
        statusDiv.innerText = "Failed to load initial data. Using defaults.";
        console.error("Error loading initial data:", error);
    }
}

// --- FUNZIONI DI DISEGNO E LOGICA CANVAS ---

function drawBase() {
    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    ctx.beginPath();
    ctx.arc(CENTER, CENTER, RADIUS_PX, 0, 2 * Math.PI);
    ctx.fillStyle = "rgba(128, 128, 128, 0.3)";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#666";
    ctx.stroke();
    // Draw Axes
    ctx.beginPath();
    ctx.strokeStyle = "#ccc";
    ctx.moveTo(CENTER, CENTER - RADIUS_PX);
    ctx.lineTo(CENTER, CENTER + RADIUS_PX);
    ctx.moveTo(CENTER - RADIUS_PX, CENTER);
    ctx.lineTo(CENTER + RADIUS_PX, CENTER);
    ctx.stroke();
}

function renderPoints() {
    drawBase();
    storedPoints.forEach(pt => {
        drawDot(pt.xPx, pt.yPx, pt.angle);
    });
}

function drawDot(x, y, angle) {
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, 2 * Math.PI);
    ctx.fillStyle = angle > 0 ? "red" : "blue";
    ctx.fill();
    ctx.strokeStyle = "white";
    ctx.lineWidth = 1;
    ctx.stroke();
}

function getCartesian(evt) {
    const rect = canvas.getBoundingClientRect();
    const mouseX = evt.clientX - rect.left;
    const mouseY = evt.clientY - rect.top;

    const xCart = (mouseX - CENTER) / RADIUS_PX;
    const yCart = -1 * (mouseY - CENTER) / RADIUS_PX;

    return { x: xCart, y: yCart, mouseX, mouseY };
}

window.toggleAddMode = function() {
    isAddMode = !isAddMode;
    if(isAddMode) {
        canvas.style.cursor = "crosshair";
        btnAdd.classList.add("active");
        btnAdd.innerText = "Click on Canvas...";
        statusDiv.innerText = "Mode: Click on the disk to place a wedge.";
    } else {
        canvas.style.cursor = "default";
        btnAdd.classList.remove("active");
        btnAdd.innerText = "+ Add Wedge";
        statusDiv.innerText = "System Ready.";
    }
}

canvas.addEventListener('mousemove', (e) => {
    const coords = getCartesian(e);
    coordDiv.innerText = `X: ${coords.x.toFixed(3)}, Y: ${coords.y.toFixed(3)}`;
});

canvas.addEventListener('click', async (e) => {
    if (!isAddMode) return;

    const coords = getCartesian(e);

    const dist = Math.sqrt(coords.x**2 + coords.y**2);
    if (dist > 1.0) {
        alert("Please click inside the circle (Radius = 1).");
        return;
    }

    const inputAngle = prompt("Enter Frank's Angle (Number):", "1");
    if (inputAngle === null) return; 

    const angle = parseFloat(inputAngle);
    if (isNaN(angle)) {
        alert("Invalid number entered.");
        return;
    }

    const params = getParams();

    try {
        const response = await fetch('/add_point', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                x: coords.x,
                y: coords.y,
                angle: angle,
                E: params.E,
                nu: params.nu,
                t: params.t
            })
        });
        const res = await response.json();
        
        if (res.status === 'success') {
            storedPoints.push({ xPx: coords.mouseX, yPx: coords.mouseY, angle: angle });
            renderPoints();
            statusDiv.innerText = `Added point and saved parameters.`;
            toggleAddMode();
        } else {
            alert("Error saving data: " + res.message);
        }
    } catch (err) {
        console.error(err);
        alert("Server error.");
    }
});

window.resetData = async function() {
    if(!confirm("Are you sure you want to clear all points?")) return;

    try {
        const response = await fetch('/reset', { method: 'POST' });
        const res = await response.json();
        if (res.status === 'success') {
            storedPoints = [];
            renderPoints(); 
            statusDiv.innerText = "Points cleared. Parameters retained.";
            outputWindow.innerText = "Waiting for computation...";
        }
    } catch (err) {
        alert("Error resetting data.");
    }
}

window.runCompute = async function() {
    statusDiv.innerText = "Running Simulation (Demo.py)... Please wait.";
    outputWindow.innerText = "Executing Demo.py...\n";

    try {
        const response = await fetch('/compute', { method: 'POST' });
        const res = await response.json();
        
        if (res.status === 'success') {
            statusDiv.innerText = "✅ Simulation Complete!";
            outputWindow.innerText += "--- Simulation Output ---\n";
            outputWindow.innerText += res.output; 
            outputWindow.scrollTop = outputWindow.scrollHeight;
            
            // Ricarica gli iframe per mostrare il nuovo plot
            plotFrames.forEach(frame => {
                // Ricarica forzato dell'iframe
                const src = frame.src.split('?')[0];
                frame.src = src + '?timestamp=' + new Date().getTime(); 
            });

        } else {
            statusDiv.innerText = "❌ Simulation Failed.";
            outputWindow.innerText += "\n--- ERROR ---\n";
            outputWindow.innerText += res.details;
            outputWindow.scrollTop = outputWindow.scrollHeight;
        }
    } catch (err) {
        statusDiv.innerText = "Connection Error.";
        outputWindow.innerText += "\n[FATAL] Failed to communicate with the Flask server.";
    }
}

// Chiamata iniziale per aggiornare i display dei parametri
updateParam('E');
updateParam('nu');
updateParam('t');