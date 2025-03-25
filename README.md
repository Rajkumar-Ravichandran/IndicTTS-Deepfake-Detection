# IndicTTS Deepfake Detection - Memory Efficient Implementation
This repository contains a lightweight, memory-efficient implementation for the IndicTTS Deepfake Detection Challenge, designed to detect synthetic speech in multilingual Indian audio samples.

**Overview**
This code provides a complete pipeline for training and evaluating a deep learning model to distinguish between real human speech and artificially generated speech (deepfakes) across multiple Indian languages. The implementation is specifically designed to run on hardware with limited memory resources.

**Key Features**

- **Memory-Efficient Design**: Minimizes memory usage through chunked processing and optimization techniques
- **Cross-Validation**: 5-fold stratified cross-validation for robust model training
- **Multi-Feature Audio Analysis**: Combines mel spectrograms, MFCCs, and zero-crossing rate
- **Lightweight Model Architecture**: Compact CNN with separate branches for different audio features

**Technical Details**
- Audio Processing
  - Fixed-length audio samples (configurable, default 5 seconds)
  - 16kHz resampling
  - Feature extraction via AudioFeatureExtractor class

- Model Architecture
  - Dual-branch CNN approach
  - Instance normalization for small batch compatibility
  - Adaptive pooling for variable-length inputs

- Training Process
  - Binary cross-entropy loss
  - Adam optimizer with learning rate scheduling
  - ROC AUC as primary evaluation metric

**Data Chunking Approach**

The implementation processes data in chunks for several critical reasons:
- **Memory Constraint Management**: Avoids loading the entire dataset at once
- **Feature Extraction Overhead**: Handles the memory multiplication that occurs during audio feature extraction
- **Controlled Memory Release**: Explicitly frees memory between chunks
- **Hardware Flexibility**: Adjustable chunk size parameter for different environments
- **GPU Memory Optimization**: Prevents CUDA out-of-memory errors during training

**Usage**

The main execution flow:
1. Loads the IndicTTS dataset from Hugging Face
2. Processes audio data in chunks to extract features
3. Performs k-fold cross-validation training
4. Makes predictions on the test set
5. Generates a submission file and visualizations

**Results**

The training process outputs:
- Submission CSV file with predictions
- Training history plots for each fold
- Saved model weights

**Requirements**
1. PyTorch & torchaudio
2. pandas & numpy
3. scikit-learn
4. matplotlib & seaborn
5. HuggingFace datasets library
6. tqdm

**License**

MIT
