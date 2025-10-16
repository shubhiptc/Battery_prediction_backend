# Battery Capacity Prediction - Backend API Documentation

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_backend.txt
```

### 2. Run the API Server
```bash
uvicorn backend_api:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access API
- **API Base URL**: `http://localhost:8000`
- **Interactive Docs**: `http://localhost:8000/docs`
- **Alternative Docs**: `http://localhost:8000/redoc`

---

## 📋 API Endpoints

### **1. Upload Training Data**
**Endpoint**: `POST /upload-training-data`

**Description**: Upload multiple CSV files for training

**Request**:
- Method: POST
- Content-Type: `multipart/form-data`
- Body: Multiple CSV files

**File Naming Convention**:
```
3500_mAh_Cell.csv
3600_mAh_Cell.csv
3700_mAh_Cell.csv
...
4900_mAh_Cell.csv
```

**Response**:
```json
{
  "status": "success",
  "message": "Uploaded 15 CSV files",
  "uploaded_files": ["3500_mAh_Cell.csv", "3600_mAh_Cell.csv", ...],
  "skipped_files": [],
  "ready_for_training": true
}
```

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/upload-training-data" \
  -F "files=@3500_mAh_Cell.csv" \
  -F "files=@3600_mAh_Cell.csv" \
  -F "files=@3700_mAh_Cell.csv"
```

**JavaScript Example**:
```javascript
const formData = new FormData();
files.forEach(file => formData.append('files', file));

const response = await fetch('http://localhost:8000/upload-training-data', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result);
```

---

### **2. Get Available Models**
**Endpoint**: `GET /available-models`

**Description**: Get list of all ML models available for training

**Response**:
```json
{
  "status": "success",
  "models": [
    "Random Forest",
    "XGBoost",
    "Support Vector Machine",
    "Gradient Boosting",
    "K-Nearest Neighbors",
    "Decision Tree",
    "Neural Network (MLP)",
    "LightGBM",
    "CatBoost"
  ],
  "count": 9
}
```

**JavaScript Example**:
```javascript
const response = await fetch('http://localhost:8000/available-models');
const data = await response.json();

// Populate dropdown with models
data.models.forEach(model => {
  const option = document.createElement('option');
  option.value = model;
  option.textContent = model;
  modelDropdown.appendChild(option);
});
```

---

### **3. Train Model**
**Endpoint**: `POST /train`

**Description**: Start training the selected model on uploaded data

**Request**:
```json
{
  "model_name": "Random Forest"
}
```

**Response** (Immediate):
```json
{
  "status": "started",
  "message": "Training Random Forest started in background",
  "model_name": "Random Forest",
  "check_status_at": "/training-status"
}
```

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"model_name": "Random Forest"}'
```

**JavaScript Example**:
```javascript
const response = await fetch('http://localhost:8000/train', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ model_name: 'Random Forest' })
});

const result = await response.json();
console.log(result);

// Start polling for status
checkTrainingStatus();
```

---

### **4. Check Training Status**
**Endpoint**: `GET /training-status`

**Description**: Get current training progress and status

**Response** (While Training):
```json
{
  "is_training": true,
  "progress": 45,
  "message": "Training model...",
  "model_name": "Random Forest",
  "result": null
}
```

**Response** (Training Complete):
```json
{
  "is_training": false,
  "progress": 100,
  "message": "Training completed successfully!",
  "model_name": "Random Forest",
  "result": {
    "status": "success",
    "model_name": "Random Forest",
    "train_accuracy": 99.58,
    "test_accuracy": 96.67,
    "training_time": "3.5 minutes",
    "model_saved": "random_forest_model.pkl",
    "per_class_accuracy": {
      "3500": 95.0,
      "3600": 100.0,
      "3700": 95.0,
      "3800": 100.0,
      "3900": 95.0,
      "4000": 100.0,
      "4100": 95.0,
      "4200": 100.0,
      "4300": 95.0,
      "4400": 100.0,
      "4500": 95.0,
      "4600": 100.0,
      "4700": 95.0,
      "4800": 100.0,
      "4900": 100.0
    }
  }
}
```

**JavaScript Example** (Polling):
```javascript
async function checkTrainingStatus() {
  const response = await fetch('http://localhost:8000/training-status');
  const status = await response.json();
  
  console.log(`Progress: ${status.progress}% - ${status.message}`);
  
  if (status.is_training) {
    // Update progress bar
    progressBar.style.width = `${status.progress}%`;
    progressText.textContent = status.message;
    
    // Poll again after 2 seconds
    setTimeout(checkTrainingStatus, 2000);
  } else if (status.result) {
    // Training complete
    if (status.result.status === 'success') {
      console.log('Training completed!');
      console.log(`Train Accuracy: ${status.result.train_accuracy}%`);
      console.log(`Test Accuracy: ${status.result.test_accuracy}%`);
      displayResults(status.result);
    } else {
      console.error('Training failed:', status.result.message);
    }
  }
}
```

---

### **5. Predict Capacity and SOH**
**Endpoint**: `POST /predict`

**Description**: Predict battery capacity and SOH% from NA file

**Request**:
- Method: POST
- Content-Type: `multipart/form-data`
- Body: Single CSV file (NA file)

**Response**:
```json
{
  "capacity": "3500 mAh",
  "soh": 71.43,
  "confidence": 95.5
}
```

**cURL Example**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@NA_A_1.csv"
```

**JavaScript Example**:
```javascript
const formData = new FormData();
formData.append('file', naFile);

const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
});

const prediction = await response.json();

console.log(`Capacity: ${prediction.capacity}`);
console.log(`SOH: ${prediction.soh}%`);
console.log(`Confidence: ${prediction.confidence}%`);

// Display to user
resultDiv.innerHTML = `
  <h3>Prediction Results</h3>
  <p><strong>Capacity:</strong> ${prediction.capacity}</p>
  <p><strong>SOH:</strong> ${prediction.soh}%</p>
  <p><strong>Confidence:</strong> ${prediction.confidence}%</p>
`;
```

---

### **6. List Trained Models**
**Endpoint**: `GET /models`

**Description**: Get list of all trained models with their information

**Response**:
```json
{
  "status": "success",
  "models": [
    {
      "model_name": "Random Forest",
      "trained_date": "2024-01-15T10:30:00",
      "train_accuracy": 99.58,
      "test_accuracy": 96.67,
      "file_path": "trained_models/random_forest_model.pkl"
    },
    {
      "model_name": "XGBoost",
      "trained_date": "2024-01-15T11:00:00",
      "train_accuracy": 99.75,
      "test_accuracy": 97.5,
      "file_path": "trained_models/xgboost_model.pkl"
    }
  ],
  "count": 2
}
```

---

### **7. Get Model Info**
**Endpoint**: `GET /model-info/{model_name}`

**Description**: Get detailed information about a specific model

**Example**: `GET /model-info/Random Forest`

**Response**:
```json
{
  "status": "success",
  "model_name": "Random Forest",
  "trained_date": "2024-01-15T10:30:00",
  "train_accuracy": 99.58,
  "test_accuracy": 96.67,
  "per_class_accuracy": {
    "3500": 95.0,
    "3600": 100.0,
    "3700": 95.0,
    ...
  },
  "training_samples": 240,
  "test_samples": 60
}
```

---

### **8. Clear Uploads**
**Endpoint**: `DELETE /clear-uploads`

**Description**: Clear all uploaded CSV files from temp directory

**Response**:
```json
{
  "status": "success",
  "message": "Cleared 15 uploaded files"
}
```

---

### **9. Health Check**
**Endpoint**: `GET /health`

**Description**: Check if API is running

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00"
}
```

---

## 🔄 Complete Frontend Integration Workflow

### **Workflow 1: Upload Training Data and Train Model**

```javascript
// Step 1: Upload training data
async function uploadTrainingData(files) {
  const formData = new FormData();
  files.forEach(file => formData.append('files', file));
  
  const response = await fetch('http://localhost:8000/upload-training-data', {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
}

// Step 2: Get available models for dropdown
async function getAvailableModels() {
  const response = await fetch('http://localhost:8000/available-models');
  const data = await response.json();
  return data.models;
}

// Step 3: Train selected model
async function trainModel(modelName) {
  const response = await fetch('http://localhost:8000/train', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_name: modelName })
  });
  
  return await response.json();
}

// Step 4: Poll training status
async function pollTrainingStatus(callback) {
  const response = await fetch('http://localhost:8000/training-status');
  const status = await response.json();
  
  callback(status);
  
  if (status.is_training) {
    setTimeout(() => pollTrainingStatus(callback), 2000);
  }
}

// Complete flow
async function completeTrainingFlow() {
  // 1. User uploads files
  const uploadResult = await uploadTrainingData(selectedFiles);
  console.log('Upload:', uploadResult);
  
  // 2. Populate model dropdown
  const models = await getAvailableModels();
  populateModelDropdown(models);
  
  // 3. User selects model and clicks train
  const trainResult = await trainModel(selectedModel);
  console.log('Training started:', trainResult);
  
  // 4. Poll for status
  pollTrainingStatus((status) => {
    updateProgressBar(status.progress);
    updateStatusMessage(status.message);
    
    if (!status.is_training && status.result) {
      if (status.result.status === 'success') {
        displayTrainingResults(status.result);
      } else {
        showError(status.result.message);
      }
    }
  });
}
```

---

### **Workflow 2: Predict Battery Capacity**

```javascript
async function predictBattery(naFile) {
  const formData = new FormData();
  formData.append('file', naFile);
  
  const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
  });
  
  const prediction = await response.json();
  
  // Display results
  displayPrediction(prediction);
  
  return prediction;
}

function displayPrediction(prediction) {
  const resultHTML = `
    <div class="prediction-result">
      <h3>🔋 Battery Analysis</h3>
      <div class="metric">
        <label>Predicted Capacity:</label>
        <value>${prediction.capacity}</value>
      </div>
      <div class="metric">
        <label>State of Health (SOH):</label>
        <value>${prediction.soh}%</value>
        <status>${getSOHStatus(prediction.soh)}</status>
      </div>
      <div class="metric">
        <label>Confidence:</label>
        <value>${prediction.confidence}%</value>
      </div>
    </div>
  `;
  
  document.getElementById('results').innerHTML = resultHTML;
}

function getSOHStatus(soh) {
  if (soh >= 80) return '🟢 Good';
  if (soh >= 60) return '🟡 Moderate';
  return '🔴 Poor';
}
```

---

## 📱 Complete HTML Example

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Battery Capacity Prediction</title>
    <style>
        .container { max-width: 800px; margin: 50px auto; padding: 20px; }
        .section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
        .button { padding: 10px 20px; margin: 10px 5px; cursor: pointer; }
        .progress-bar { width: 100%; height: 30px; background: #f0f0f0; border-radius: 5px; overflow: hidden; }
        .progress-fill { height: 100%; background: #4CAF50; transition: width 0.3s; }
        .result { padding: 15px; margin: 10px 0; background: #e8f5e9; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔋 Battery Capacity Prediction System</h1>
        
        <!-- Section 1: Training -->
        <div class="section">
            <h2>📚 Training Section</h2>
            
            <div>
                <h3>Step 1: Upload Training Data</h3>
                <input type="file" id="trainingFiles" multiple accept=".csv">
                <button class="button" onclick="uploadFiles()">Upload Files</button>
                <div id="uploadStatus"></div>
            </div>
            
            <div style="margin-top: 20px;">
                <h3>Step 2: Select Model</h3>
                <select id="modelSelect">
                    <option value="">Loading models...</option>
                </select>
            </div>
            
            <div style="margin-top: 20px;">
                <h3>Step 3: Train Model</h3>
                <button class="button" onclick="startTraining()">🚀 Train Model</button>
            </div>
            
            <div id="trainingProgress" style="display: none; margin-top: 20px;">
                <h3>Training Progress</h3>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <p id="progressMessage"></p>
            </div>
            
            <div id="trainingResults" style="display: none;"></div>
        </div>
        
        <!-- Section 2: Prediction -->
        <div class="section">
            <h2>🔮 Prediction Section</h2>
            
            <div>
                <h3>Upload Battery Data (NA File)</h3>
                <input type="file" id="predictionFile" accept=".csv">
                <button class="button" onclick="predictCapacity()">Predict</button>
            </div>
            
            <div id="predictionResults" style="display: none;"></div>
        </div>
    </div>

    <script>
        const API_URL = 'http://localhost:8000';
        
        // Load available models on page load
        window.onload = async () => {
            const response = await fetch(`${API_URL}/available-models`);
            const data = await response.json();
            
            const select = document.getElementById('modelSelect');
            select.innerHTML = '<option value="">Select a model...</option>';
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                select.appendChild(option);
            });
        };
        
        // Upload training files
        async function uploadFiles() {
            const fileInput = document.getElementById('trainingFiles');
            const files = fileInput.files;
            
            if (files.length === 0) {
                alert('Please select CSV files to upload');
                return;
            }
            
            const formData = new FormData();
            for (let file of files) {
                formData.append('files', file);
            }
            
            try {
                const response = await fetch(`${API_URL}/upload-training-data`, {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                document.getElementById('uploadStatus').innerHTML = `
                    <div class="result">
                        ✅ ${result.message}<br>
                        Uploaded: ${result.uploaded_files.length} files
                    </div>
                `;
            } catch (error) {
                alert('Upload failed: ' + error.message);
            }
        }
        
        // Start training
        async function startTraining() {
            const modelSelect = document.getElementById('modelSelect');
            const modelName = modelSelect.value;
            
            if (!modelName) {
                alert('Please select a model');
                return;
            }
            
            try {
                const response = await fetch(`${API_URL}/train`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model_name: modelName })
                });
                
                const result = await response.json();
                
                if (result.status === 'started') {
                    document.getElementById('trainingProgress').style.display = 'block';
                    pollTrainingStatus();
                }
            } catch (error) {
                alert('Training failed to start: ' + error.message);
            }
        }
        
        // Poll training status
        async function pollTrainingStatus() {
            const response = await fetch(`${API_URL}/training-status`);
            const status = await response.json();
            
            // Update progress bar
            document.getElementById('progressFill').style.width = `${status.progress}%`;
            document.getElementById('progressMessage').textContent = status.message;
            
            if (status.is_training) {
                setTimeout(pollTrainingStatus, 2000);
            } else if (status.result) {
                if (status.result.status === 'success') {
                    displayTrainingResults(status.result);
                } else {
                    alert('Training failed: ' + status.result.message);
                }
            }
        }
        
        // Display training results
        function displayTrainingResults(result) {
            document.getElementById('trainingResults').style.display = 'block';
            document.getElementById('trainingResults').innerHTML = `
                <div class="result">
                    <h3>✅ Training Completed!</h3>
                    <p><strong>Model:</strong> ${result.model_name}</p>
                    <p><strong>Training Accuracy:</strong> ${result.train_accuracy}%</p>
                    <p><strong>Testing Accuracy:</strong> ${result.test_accuracy}%</p>
                    <p><strong>Training Time:</strong> ${result.training_time}</p>
                    <p><strong>Model Saved:</strong> ${result.model_saved}</p>
                </div>
            `;
        }
        
        // Predict capacity
        async function predictCapacity() {
            const fileInput = document.getElementById('predictionFile');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select a CSV file');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch(`${API_URL}/predict`, {
                    method: 'POST',
                    body: formData
                });
                
                const prediction = await response.json();
                
                const sohStatus = prediction.soh >= 80 ? '🟢 Good' : 
                                 prediction.soh >= 60 ? '🟡 Moderate' : '🔴 Poor';
                
                document.getElementById('predictionResults').style.display = 'block';
                document.getElementById('predictionResults').innerHTML = `
                    <div class="result">
                        <h3>🔋 Prediction Results</h3>
                        <p><strong>Predicted Capacity:</strong> ${prediction.capacity}</p>
                        <p><strong>State of Health (SOH):</strong> ${prediction.soh}% ${sohStatus}</p>
                        <p><strong>Confidence:</strong> ${prediction.confidence}%</p>
                    </div>
                `;
            } catch (error) {
                alert('Prediction failed: ' + error.message);
            }
        }
    </script>
</body>
</html>
```

---

## 🐛 Error Handling

### Common Errors and Solutions

**Error**: `No trained model found`
- **Solution**: Train a model first using `/train` endpoint

**Error**: `No training data uploaded`
- **Solution**: Upload CSV files using `/upload-training-data` first

**Error**: `Training already in progress`
- **Solution**: Wait for current training to complete or restart the API

**Error**: `Model 'XYZ' not available`
- **Solution**: Check available models using `/available-models` endpoint

**Error**: `CSV must contain columns`
- **Solution**: Ensure CSV has: Frequency (Hz), Amplitude S12 Sample 1, Phase S12 Sample 1

---

## 📊 API Response Status Codes

- **200**: Success
- **400**: Bad Request (invalid input)
- **404**: Not Found (model/resource not found)
- **500**: Internal Server Error

---

## 🔒 Security Notes (For Production)

1. **CORS**: Update `allow_origins` to your frontend domain
2. **File Size**: Add file size limits
3. **Authentication**: Add API key or JWT authentication
4. **Rate Limiting**: Implement rate limiting
5. **Input Validation**: Add more robust input validation

---

## 💡 Tips for Frontend Developers

1. **Always check training status** before allowing prediction
2. **Show progress bar** during training for better UX
3. **Handle errors gracefully** with user-friendly messages
4. **Cache model list** to avoid repeated API calls
5. **Add loading indicators** during API calls
6. **Validate file format** on frontend before uploading

---

## 📞 Support

For issues or questions:
- Check API documentation: `/docs`
- Check health status: `/health`
- Review logs in terminal where API is running