const imageInput = document.getElementById('imageInput');
const dropZone = document.getElementById('dropZone');
const preview = document.getElementById('preview');
const submitBtn = document.getElementById('submitBtn');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');

let selectedFile = null;

function setStatus(message) {
	statusEl.textContent = message || '';
}

function handleFiles(files) {
	const file = files && files[0];
	if (!file) return;

	if (!file.type || !file.type.startsWith('image/')) {
		alert('Please select an image file.');
		return;
	}

	selectedFile = file;
	setStatus(`Selected: ${file.name} (${Math.round(file.size / 1024)} KB)`);

	const reader = new FileReader();
	reader.onload = (e) => {
		preview.src = e.target.result;
		preview.style.display = 'block';
		submitBtn.disabled = false;
	};
	reader.onerror = () => {
		setStatus('Failed to read file');
	};
	reader.readAsDataURL(file);
}

// File input change
imageInput?.addEventListener('change', (e) => handleFiles(e.target.files));

// Drag & drop
['dragenter', 'dragover'].forEach((evt) => {
	dropZone?.addEventListener(evt, (e) => {
		e.preventDefault();
		e.stopPropagation();
		dropZone.classList.add('hover');
	});
});

['dragleave', 'drop'].forEach((evt) => {
	dropZone?.addEventListener(evt, (e) => {
		e.preventDefault();
		e.stopPropagation();
		dropZone.classList.remove('hover');
	});
});

dropZone?.addEventListener('drop', (e) => {
	const files = e.dataTransfer?.files;
	handleFiles(files);
});

// Submit handler (stub for inference)
submitBtn?.addEventListener('click', async () => {
	if (!selectedFile) return;
	setStatus('Submitting image…');

	//Sending the image data from frontend to backend for inference

	// Example: If you add a backend, you can POST the file
	const formData = new FormData();
	formData.append('image', selectedFile);
	const resp = await fetch('http://127.0.0.1:5000/predict', { method: 'POST', body: formData });
	const data = await resp.json();
	resultEl.textContent = `Prediction: ${data.prediction}
	, Index: ${data.predicted_index}`;

	// For now, just show file details
	// resultEl.textContent = `Ready for inference: ${selectedFile.name}`;
	setStatus('');
});

