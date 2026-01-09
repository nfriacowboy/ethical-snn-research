#!/bin/bash
# ROCm Installation Script for AMD GPUs

echo "Installing ROCm for AMD GPU support..."

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Error: This script is only for Linux systems"
    exit 1
fi

# Add ROCm repository
echo "Adding ROCm repository..."
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

# Update package list
sudo apt update

# Install ROCm
echo "Installing ROCm..."
sudo apt install -y rocm-dkms

# Add user to render and video groups
sudo usermod -a -G render,video $USER

# Install PyTorch with ROCm support
echo "Installing PyTorch with ROCm support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

echo ""
echo "ROCm installation complete!"
echo "Please reboot your system for changes to take effect."
echo "After reboot, run 'verify_gpu.py' to test GPU availability."
