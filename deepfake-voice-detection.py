# IndicTTS Deepfake Detection Challenge - Memory Efficient Implementation
# Multilingual Indian Speech Deepfake Detection with Hugging Face Dataset

import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
from datasets import load_dataset, Audio
import warnings
warnings.filterwarnings("ignore")

# Set seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
   
set_seed()

# Check for available GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define output path
OUTPUT_DIR = './'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature Extraction Functions (lightweight version)
class AudioFeatureExtractor:
    def __init__(self, sr=16000, n_mels=64, n_fft=1024, hop_length=512):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
       
        # Feature extractors
        self.melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=20,  # Reduced from 40 to save memory
            melkwargs={
                'n_fft': n_fft,
                'n_mels': n_mels,
                'hop_length': hop_length
            }
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
       
    def extract_features(self, waveform):
        """Extract multiple features from audio waveform"""
        # Ensure waveform is 2D [channels, time]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
           
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
       
        # Extract Mel Spectrogram
        mel_spec = self.melspec_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
       
        # Extract MFCCs
        mfcc = self.mfcc_transform(waveform)
       
        # Calculate Zero Crossing Rate (simplified)
        zcr = torch.sum(torch.abs(torch.sign(waveform[:, 1:]) - torch.sign(waveform[:, :-1])), dim=1) / (2 * (waveform.shape[1] - 1))
       
        return {
            'mel_spec_db': mel_spec_db,
            'mfcc': mfcc,
            'zcr': zcr,
        }

# Custom Dataset for efficient processing
class EfficientAudioDataset(Dataset):
    def __init__(self, hf_dataset, subset_indices=None, max_duration=5, sample_rate=16000, is_test=False):
        self.hf_dataset = hf_dataset
        self.subset_indices = subset_indices if subset_indices is not None else range(len(hf_dataset))
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_length = max_duration * sample_rate
        self.is_test = is_test
        self.feature_extractor = AudioFeatureExtractor(sr=sample_rate)
       
    def __len__(self):
        return len(self.subset_indices)
   
    def __getitem__(self, idx):
        item_idx = self.subset_indices[idx]
        item = self.hf_dataset[item_idx]
       
        # Get audio data
        audio_array = item['audio']['array']
        audio_sampling_rate = item['audio']['sampling_rate']
       
        # Convert numpy array to torch tensor
        waveform = torch.from_numpy(audio_array).float()
       
        # Process waveform
        try:
            # Resample if needed
            if audio_sampling_rate != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, audio_sampling_rate, self.sample_rate)
           
            # Ensure waveform is 2D [channels, time]
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
               
            # Pad or truncate to fixed length
            if waveform.shape[1] < self.max_length:
                # Pad
                padding = self.max_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            else:
                # Truncate
                waveform = waveform[:, :self.max_length]
               
            # Extract features
            features = self.feature_extractor.extract_features(waveform)
           
            # Free memory
            del waveform
            torch.cuda.empty_cache()
           
            # Get target (for test set, use 0 as placeholder)
            if self.is_test:
                target = torch.tensor(0.0)
            else:
                target = torch.tensor(float(item['is_tts']))
               
            return {
                'id': item['id'],
                'features': features,
                'target': target,
                'language': item['language']
            }
           
        except Exception as e:
            print(f"Error processing audio sample {item['id']}: {e}")
            # Return zeros as fallback
            features = {
                'mel_spec_db': torch.zeros((1, self.feature_extractor.n_mels,
                                            self.max_length // self.feature_extractor.hop_length + 1)),
                'mfcc': torch.zeros((1, 20, self.max_length // self.feature_extractor.hop_length + 1)),
                'zcr': torch.zeros(1),
            }
           
            if self.is_test:
                target = torch.tensor(0.0)
            else:
                target = torch.tensor(float(item['is_tts']) if 'is_tts' in item else 0.0)
               
            return {
                'id': item['id'],
                'features': features,
                'target': target,
                'language': item['language']
            }

# Feature Collation Function for Batch Processing
def collate_fn(batch):
    ids = [item['id'] for item in batch]
    languages = [item['language'] for item in batch]
    targets = torch.stack([item['target'] for item in batch])
   
    # Process mel spectrograms
    mel_specs = torch.stack([item['features']['mel_spec_db'] for item in batch])
   
    # Process MFCCs
    mfccs = torch.stack([item['features']['mfcc'] for item in batch])
   
    # Process ZCR
    zcrs = torch.stack([item['features']['zcr'] for item in batch])
   
    return {
        'ids': ids,
        'languages': languages,
        'mel_specs': mel_specs,
        'mfccs': mfccs,
        'zcrs': zcrs,
        'targets': targets
    }

# Lighter Neural Network Architecture (BatchNorm-free for small batches)
class LightDeepfakeDetectionModel(nn.Module):
    def __init__(self, n_mels=64, n_mfcc=20):
        super(LightDeepfakeDetectionModel, self).__init__()
       
        # Mel Spectrogram processing branch - using Instance Normalization instead of Batch Normalization
        self.mel_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.mel_in1 = nn.InstanceNorm2d(16)  # Instance norm works with batch size 1
        self.mel_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.mel_in2 = nn.InstanceNorm2d(32)
       
        # MFCC processing branch
        self.mfcc_conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.mfcc_in1 = nn.InstanceNorm2d(16)
       
        # Shared layers
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
       
        # Global pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
       
        # Fully connected layers for combined features
        self.fc1 = nn.Linear(32 + 16 + 1, 64)  # mel + mfcc + zcr
        self.layer_norm = nn.LayerNorm(64)  # Layer norm instead of batch norm
        self.fc2 = nn.Linear(64, 1)
       
    def forward(self, mel_specs, mfccs, zcrs):
        # Process Mel Spectrogram
        x_mel = self.relu(self.mel_in1(self.mel_conv1(mel_specs)))
        x_mel = self.pool(x_mel)
        x_mel = self.relu(self.mel_in2(self.mel_conv2(x_mel)))
        x_mel = self.pool(x_mel)
        x_mel = self.adaptive_pool(x_mel)
        x_mel = x_mel.view(x_mel.size(0), -1)  # Flatten
       
        # Process MFCCs
        x_mfcc = self.relu(self.mfcc_in1(self.mfcc_conv1(mfccs)))
        x_mfcc = self.pool(x_mfcc)
        x_mfcc = self.adaptive_pool(x_mfcc)
        x_mfcc = x_mfcc.view(x_mfcc.size(0), -1)  # Flatten
       
        # Process ZCR
        zcrs_mean = zcrs.mean(dim=1).unsqueeze(1)
       
        # Concatenate all features
        combined = torch.cat([x_mel, x_mfcc, zcrs_mean], dim=1)
       
        # Fully connected layers
        x = self.fc1(combined)
        x = self.layer_norm(x)  # LayerNorm works with any batch size
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
       
        return torch.sigmoid(x)

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, scheduler=None):
    best_val_auc = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': []
    }
   
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
       
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in progress_bar:
            # Move batch to device
            mel_specs = batch['mel_specs'].to(device)
            mfccs = batch['mfccs'].to(device)
            zcrs = batch['zcrs'].to(device)
            targets = batch['targets'].to(device).unsqueeze(1)
           
            # Forward pass
            optimizer.zero_grad()
            outputs = model(mel_specs, mfccs, zcrs)
            loss = criterion(outputs, targets)
           
            # Backward pass
            loss.backward()
            optimizer.step()
           
            # Update metrics
            train_loss += loss.item()
            train_batches += 1
            progress_bar.set_postfix({'loss': loss.item()})
           
            # Clean up GPU memory
            del mel_specs, mfccs, zcrs, targets, outputs
            torch.cuda.empty_cache()
       
        # Calculate average loss for the epoch
        train_loss /= train_batches
        history['train_loss'].append(train_loss)
       
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        val_batches = 0
       
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch in progress_bar:
                # Move batch to device
                mel_specs = batch['mel_specs'].to(device)
                mfccs = batch['mfccs'].to(device)
                zcrs = batch['zcrs'].to(device)
                targets = batch['targets'].to(device).unsqueeze(1)
               
                # Forward pass
                outputs = model(mel_specs, mfccs, zcrs)
                loss = criterion(outputs, targets)
               
                # Update metrics
                val_loss += loss.item()
                val_batches += 1
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
               
                progress_bar.set_postfix({'loss': loss.item()})
               
                # Clean up GPU memory
                del mel_specs, mfccs, zcrs, targets, outputs
                torch.cuda.empty_cache()
       
        # Calculate average validation loss
        val_loss /= val_batches
        history['val_loss'].append(val_loss)
       
        # Calculate ROC AUC
        val_auc = roc_auc_score(val_targets, val_preds)
        history['val_auc'].append(val_auc)
       
        # Step the scheduler if provided
        if scheduler:
            scheduler.step(val_loss)
       
        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val AUC: {val_auc:.4f}')
       
        # Save the best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            print(f'  New best model saved with AUC: {val_auc:.4f}')
   
    return history

# Inference Function with Memory Management
def predict(model, test_loader, device):
    model.eval()
    predictions = {}
   
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Testing')
        for batch in progress_bar:
            # Move batch to device
            mel_specs = batch['mel_specs'].to(device)
            mfccs = batch['mfccs'].to(device)
            zcrs = batch['zcrs'].to(device)
           
            # Get predictions
            outputs = model(mel_specs, mfccs, zcrs)
           
            # Store predictions
            for i, audio_id in enumerate(batch['ids']):
                predictions[audio_id] = outputs[i].item()
           
            # Clean up GPU memory
            del mel_specs, mfccs, zcrs, outputs
            torch.cuda.empty_cache()
   
    return predictions

# Process dataset in chunks to avoid memory issues
def process_dataset_in_chunks(hf_dataset, is_test=False, chunk_size=500):
    """Process the dataset in smaller chunks to avoid memory issues"""
    all_data = []
    targets = []
    dataset_size = len(hf_dataset)
   
    # Ensure chunk_size doesn't exceed dataset size
    chunk_size = min(chunk_size, dataset_size)
   
    # Process the dataset in chunks
    for start_idx in tqdm(range(0, dataset_size, chunk_size), desc="Processing dataset chunks"):
        end_idx = min(start_idx + chunk_size, dataset_size)
        chunk_indices = range(start_idx, end_idx)
       
                    # Create dataset for this chunk
        chunk_dataset = EfficientAudioDataset(
            hf_dataset,
            subset_indices=chunk_indices,
            max_duration=3,  # Even shorter duration to save memory
            sample_rate=16000,
            is_test=is_test
        )
       
        # Create dataloader for this chunk
        chunk_loader = DataLoader(
            chunk_dataset,
            batch_size=4,  # Very small batch size to avoid OOM
            shuffle=False,
            num_workers=0,  # No parallel workers to reduce memory usage
            collate_fn=collate_fn
        )
       
        # Process chunks
        for batch in chunk_loader:
            # Add batch data to all_data
            for i in range(len(batch['ids'])):
                item = {
                    'id': batch['ids'][i],
                    'language': batch['languages'][i],
                    'target': batch['targets'][i],
                    'features': {
                        'mel_spec_db': batch['mel_specs'][i],
                        'mfcc': batch['mfccs'][i],
                        'zcr': batch['zcrs'][i]
                    }
                }
                all_data.append(item)
                if not is_test:
                    targets.append(batch['targets'][i].item())
       
        # Clean up memory
        del chunk_dataset, chunk_loader
        torch.cuda.empty_cache()
        gc.collect()
   
    return all_data, targets

# Main execution
def main():
    print("Starting IndicTTS Deepfake Detection pipeline (Memory-Efficient Version)...")
   
    # Load the dataset from Hugging Face (with caching to disk)
    print("Loading dataset from Hugging Face...")
    try:
        dataset = load_dataset("SherryT997/IndicTTS-Deepfake-Challenge-Data")
       
        # Cast the audio column with proper settings
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
       
        # Convert to pandas DataFrames for summary
        train_df = pd.DataFrame({
            'id': dataset['train']['id'],
            'language': dataset['train']['language'],
            'is_tts': dataset['train']['is_tts']
        })
       
        test_df = pd.DataFrame({
            'id': dataset['test']['id'],
            'language': dataset['test']['language'],
        })
       
        print(f"Training samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
       
        # Display language distribution
        print("\nLanguage distribution in training data:")
        print(train_df['language'].value_counts())
       
        # Display label distribution
        print("\nLabel distribution in training data:")
        print(train_df['is_tts'].value_counts())
       
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please make sure you have an internet connection and the dataset exists.")
        return
   
    # Set up training parameters
    n_folds = 5
    BATCH_SIZE = 4   # Very small batch size to prevent OOM errors
    NUM_EPOCHS = 5   # Reduced epochs for testing
    LEARNING_RATE = 0.001
    CHUNK_SIZE = 200  # Smaller chunks to prevent memory issues
   
    # Check if processed data is already saved
    processed_data_file = os.path.join(OUTPUT_DIR, 'processed_data.pth')
   
    if os.path.exists(processed_data_file):
        print(f"Loading pre-processed data from {processed_data_file}")
        saved_data = torch.load(processed_data_file)
        train_data = saved_data['train_data']
        train_targets = saved_data['train_targets']
        test_data = saved_data['test_data']
    else:
        print("Processing dataset in chunks to avoid memory issues...")
       
        # Process training data
        print("Processing training data...")
        train_data, train_targets = process_dataset_in_chunks(dataset['train'], is_test=False, chunk_size=CHUNK_SIZE)
       
        # Process test data
        print("Processing test data...")
        test_data, _ = process_dataset_in_chunks(dataset['test'], is_test=True, chunk_size=CHUNK_SIZE)
       
        # Save processed data
        torch.save({
            'train_data': train_data,
            'train_targets': train_targets,
            'test_data': test_data
        }, processed_data_file)
        print(f"Saved processed data to {processed_data_file}")
   
    # Set up stratified K-fold
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_indices = list(kf.split(np.arange(len(train_data)), train_targets))
   
    # Train model for each fold
    fold_histories = []
    test_predictions = []
   
    for fold, (train_idx, val_idx) in enumerate(fold_indices):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/{n_folds}")
        print(f"{'='*50}")
       
        # Create train and validation subsets
        train_subset = [train_data[i] for i in train_idx]
        val_subset = [train_data[i] for i in val_idx]
       
        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_fn
        )
       
        val_loader = DataLoader(
            val_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn
        )
       
        # Initialize model
        model = LightDeepfakeDetectionModel().to(device)
       
        # Initialize optimizer and loss function
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
       
        # Train the model
        history = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            num_epochs=NUM_EPOCHS,
            scheduler=scheduler
        )
       
        fold_histories.append(history)
       
        # Create test loader
        test_loader = DataLoader(
            test_data,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn
        )
       
        # Load the best model for this fold
        model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model.pth')))
       
        # Predict on test set
        fold_predictions = predict(model, test_loader, device)
        test_predictions.append(fold_predictions)
       
        # Clear memory
        del model, optimizer, scheduler, train_loader, val_loader, test_loader
        torch.cuda.empty_cache()
        gc.collect()
   
    # Average predictions from all folds
    final_predictions = {}
    for audio_id in test_predictions[0].keys():
        final_predictions[audio_id] = np.mean([fold_pred[audio_id] for fold_pred in test_predictions])
   
    # Create submission file
    submission_df = pd.DataFrame({
        'id': list(final_predictions.keys()),
        'is_tts': list(final_predictions.values())
    })
   
    # Save submission file
    submission_df.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)
    print(f"Saved final submission to {os.path.join(OUTPUT_DIR, 'submission.csv')}")
   
    # Plot training history
    plt.figure(figsize=(15, 5))
   
    plt.subplot(1, 3, 1)
    for i, history in enumerate(fold_histories):
        plt.plot(history['train_loss'], label=f'Fold {i+1}')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
   
    plt.subplot(1, 3, 2)
    for i, history in enumerate(fold_histories):
        plt.plot(history['val_loss'], label=f'Fold {i+1}')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
   
    plt.subplot(1, 3, 3)
    for i, history in enumerate(fold_histories):
        plt.plot(history['val_auc'], label=f'Fold {i+1}')
    plt.title('Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
   
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    plt.close()
   
    print("IndicTTS Deepfake Detection pipeline completed successfully!")

main()
