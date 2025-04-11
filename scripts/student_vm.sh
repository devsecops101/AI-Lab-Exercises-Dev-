#!/bin/bash
useradd -m -s /bin/bash student
echo "student:student123" | chpasswd

# Install required packages
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv git

# Clone the repository into student's home directory
cd /home/student
git clone https://github.com/devsecops101/AI-Lab-Exercises-Dev-.git labs

# Create the destination directory for the model file
mkdir -p /home/student/.cache/torch/hub/checkpoints/

# Copy the model file to the required location
cp /home/student/labs/lab1/models/resnet18-f37072fd.pth /home/student/.cache/torch/hub/checkpoints/

# Set proper ownership and permissions
chown -R student:student /home/student/.cache
chmod -R 755 /home/student/.cache

# Set proper ownership and permissions for labs directory
chown -R student:student /home/student/labs
chmod -R 755 /home/student/labs

# Create and activate virtual environment
python3 -m venv /home/student/venv

# Install Python packages in the virtual environment
/home/student/venv/bin/pip install --no-cache-dir \
    torch \
    torchvision \
    matplotlib \
    numpy \
    pandas \
    scikit-learn \
    tensorflow \
    joblib

# Make the virtual environment accessible to the student user
chown -R student:student /home/student/venv

# Add activation to student's .bashrc
echo 'source /home/student/venv/bin/activate' >> /home/student/.bashrc 