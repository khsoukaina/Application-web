from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import os
import zipfile
import torch
import json
from torchvision import transforms, datasets
from PIL import Image, UnidentifiedImageError
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision import models
import pandas as pd
import uuid
from sklearn.model_selection import train_test_split
from openpyxl import Workbook
from openpyxl.drawing.image import Image as OpenpyxlImage

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'static/uploads'
NEW_CATEGORY_FOLDER = 'static/new_category'
MODEL_PATH = 'model_trained.pth'
CATEGORIES_FILE = 'categories.json'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['NEW_CATEGORY_FOLDER'] = NEW_CATEGORY_FOLDER

# Ensure the upload and new category folders exist
for folder in [UPLOAD_FOLDER, NEW_CATEGORY_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Load categories from the JSON file or initialize with default categories
if os.path.exists(CATEGORIES_FILE):
    with open(CATEGORIES_FILE, 'r') as f:
        CLASS_NAMES = json.load(f)
else:
    CLASS_NAMES = ["antenne", "pylone", "fh"]

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

class MultiTaskResNeXt(nn.Module):
    def __init__(self, num_classes=3):
        super(MultiTaskResNeXt, self).__init__()
        self.model = models.resnext101_32x8d(weights=models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features

        self.model.fc = nn.Identity()

        self.heads = nn.ModuleList()
        for _ in range(num_classes):
            self.heads.append(nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 2)
            ))

    def forward(self, x):
        features = self.model(x)
        outputs = [head(features) for head in self.heads]
        return outputs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, num_classes):
    model = MultiTaskResNeXt(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)

    # Rename keys in state_dict to match new model structure
    new_state_dict = {}
    for key, value in state_dict.items():
        if "pylone_fc" in key:
            new_key = key.replace("pylone_fc", "heads.0")
        elif "antenne_fc" in key:
            new_key = key.replace("antenne_fc", "heads.1")
        elif "fh_fc" in key:
            new_key = key.replace("fh_fc", "heads.2")
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Model loaded with {num_classes} classes.")
    return model

model = load_model(MODEL_PATH, len(CLASS_NAMES))

def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def classify_image(model, image_path):
    try:
        image = Image.open(image_path).convert('RGB')
    except UnidentifiedImageError:
        print(f"Error opening image: {image_path}")
        return None
    image = transform_image(image)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image)
        results = {}
        for i, output in enumerate(outputs):
            result = torch.softmax(output, dim=1)
            class_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'Class {i}'
            results[class_name] = 'OK' if result[0][1].item() > result[0][0].item() else 'NOK'
    return results

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def create_excel(results, zip_filename):
    wb = Workbook()
    ws = wb.active
    ws.append(['Image', 'Nom'] + CLASS_NAMES)
    
    for result in results:
        img_path = os.path.join(app.root_path, 'static', result['path'])
        img = OpenpyxlImage(img_path)
        ws.append(['', result['name']] + [result[category] for category in CLASS_NAMES])
        ws.add_image(img, f'A{ws.max_row}')
    
    excel_path = os.path.join(app.root_path, f'{zip_filename}_results.xlsx')
    wb.save(excel_path)
    return excel_path

@app.route('/')
def home():
    return render_template('home.html', categories=CLASS_NAMES)

@app.route('/upload', methods=['POST'])
def upload_images():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        unique_folder = str(uuid.uuid4())
        upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], unique_folder)
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        zip_path = os.path.join(upload_folder, file.filename)
        file.save(zip_path)

        extract_zip(zip_path, upload_folder)
        
        image_paths = []
        supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp', '.jfif')  # Add more extensions if needed
        for root, _, files in os.walk(upload_folder):
            for file_name in files:
                if file_name.lower().endswith(supported_extensions):
                    image_paths.append(os.path.join(root, file_name))

        results = []
        categories = CLASS_NAMES  
        for image_path in image_paths:
            classification = classify_image(model, image_path)
            if classification is None:
                continue
            relative_path = os.path.relpath(image_path, start='static')
            result = {
                'path': relative_path.replace('\\', '/'),
                'name': os.path.basename(image_path),
            }
            result.update(classification)
            results.append(result)

        zip_filename = os.path.splitext(file.filename)[0]
        excel_path = create_excel(results, zip_filename)

        return render_template('result.html', results=results, categories=categories, excel_path=excel_path)

@app.route('/add_category', methods=['GET', 'POST'])
def add_category():
    if request.method == 'POST':
        category_name = request.form['category_name']
        if category_name == "autre":
            category_name = request.form['new_category_name']

        files_ok = request.files.getlist('file_ok')
        files_nok = request.files.getlist('file_nok')
        supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp', '.jfif')
        if category_name and files_ok and files_nok:
            category_folder_ok = os.path.join(app.config['NEW_CATEGORY_FOLDER'], category_name, 'ok')
            category_folder_nok = os.path.join(app.config['NEW_CATEGORY_FOLDER'], category_name, 'nok')
            if not os.path.exists(category_folder_ok):
                os.makedirs(category_folder_ok)
            if not os.path.exists(category_folder_nok):
                os.makedirs(category_folder_nok)
            
            for file in files_ok:
                if file and file.filename.lower().endswith(supported_extensions):
                    file_path = os.path.join(category_folder_ok, file.filename)
                    file.save(file_path)
            for file in files_nok:
                if file and file.filename.lower().endswith(supported_extensions):
                    file_path = os.path.join(category_folder_nok, file.filename)
                    file.save(file_path)
            
            if category_name not in CLASS_NAMES:
                CLASS_NAMES.append(category_name)
                with open(CATEGORIES_FILE, 'w') as f:
                    json.dump(CLASS_NAMES, f)

            re_train_model(category_name, category_folder_ok, category_folder_nok)
            flash("Training completed and model updated", "success")
            return redirect(url_for('home'))
    return render_template('add_category.html', categories=CLASS_NAMES)

@app.route('/download_excel')
def download_excel():
    excel_path = request.args.get('path')
    return send_file(excel_path, as_attachment=True, download_name=os.path.basename(excel_path))

def re_train_model(new_category, ok_folder, nok_folder):
    global model
    num_classes = len(CLASS_NAMES)

    new_model = MultiTaskResNeXt(num_classes=num_classes)
    new_model.load_state_dict(model.state_dict(), strict=False)

    for layer in new_model.heads[-1]:
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    new_model.to(device)

    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=0.2),
        transforms.RandomVerticalFlip(),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_paths_ok = [os.path.join(ok_folder, img) for img in os.listdir(ok_folder) if os.path.isfile(os.path.join(ok_folder, img))]
    image_paths_nok = [os.path.join(nok_folder, img) for img in os.listdir(nok_folder) if os.path.isfile(os.path.join(nok_folder, img))]

    all_image_paths = image_paths_ok + image_paths_nok
    labels = [0] * len(image_paths_ok) + [1] * len(image_paths_nok)

    train_paths, val_paths, train_labels, val_labels = train_test_split(all_image_paths, labels, test_size=0.2, random_state=42)

    train_data = CustomImageDataset(train_paths, train_labels, transform=transform)
    val_data = CustomImageDataset(val_paths, val_labels, transform=transform)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(new_model.parameters(), lr=0.0001, weight_decay=0.01)

    for epoch in range(10):
        new_model.train()
        running_loss = 0.0
        for inputs, batch_labels in train_loader:
            inputs, batch_labels = inputs.to(device), batch_labels.to(device)
            optimizer.zero_grad()

            outputs = new_model(inputs)
            loss = criterion(outputs[-1], batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader.dataset):.4f}')

        new_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, batch_labels in val_loader:
                inputs, batch_labels = inputs.to(device), batch_labels.to(device)
                outputs = new_model(inputs)
                loss = criterion(outputs[-1], batch_labels)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        print(f'Epoch {epoch+1}, Validation Loss: {val_loss:.4f}')

    torch.save(new_model.state_dict(), MODEL_PATH)
    model = load_model(MODEL_PATH, num_classes)
    print("Model updated and saved.")

if __name__ == '__main__':
    app.run(debug=True)
