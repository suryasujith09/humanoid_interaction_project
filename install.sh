#!/bin/bash
# Installation script for Humanoid Interaction Project
# Save as: ~/humanoid_interaction_project/install.sh

set -e  # Exit on error

echo "=========================================="
echo "Humanoid Interaction System Installation"
echo "=========================================="
echo ""

PROJECT_DIR="$HOME/humanoid_interaction_project"

# Check if running from correct directory
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Error: Project directory not found at $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"

echo "Step 1: Creating directory structure..."
mkdir -p scripts/controllers
mkdir -p scripts/triggers
mkdir -p scripts/utils
mkdir -p actions/custom
mkdir -p config
mkdir -p tests
mkdir -p logs

echo "Step 2: Creating __init__.py files..."
touch scripts/__init__.py
touch scripts/controllers/__init__.py
touch scripts/triggers/__init__.py
touch scripts/utils/__init__.py

echo "Step 3: Setting executable permissions..."
chmod +x scripts/*.py
chmod +x scripts/controllers/*.py
chmod +x scripts/triggers/*.py
chmod +x scripts/utils/*.py

echo "Step 4: Installing Python dependencies..."
# Check if running in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    PIP_CMD="pip3"
else
    PIP_CMD="pip"
fi

$PIP_CMD install --user pyyaml

echo "Step 5: Verifying ROS installation..."
if [ -z "$ROS_DISTRO" ]; then
    echo "Warning: ROS not sourced. Please run:"
    echo "  source /opt/ros/noetic/setup.bash"
    echo "  source ~/ros_ws/devel/setup.bash"
else
    echo "ROS $ROS_DISTRO detected ?"
fi

echo "Step 6: Verifying AiNex controller..."
if [ -d "$HOME/software/ainex_controller" ]; then
    echo "AiNex controller found ?"
else
    echo "Warning: AiNex controller not found at ~/software/ainex_controller"
fi

echo "Step 7: Checking serial port permissions..."
if [ -e "/dev/ttyAMA0" ]; then
    echo "Serial port /dev/ttyAMA0 found ?"
    groups | grep -q dialout && echo "User in dialout group ?" || echo "Warning: User not in dialout group"
else
    echo "Warning: Serial port /dev/ttyAMA0 not found"
fi

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Copy your code files to the appropriate directories:"
echo "   - config/robot_config.yaml"
echo "   - scripts/utils/logger.py"
echo "   - scripts/controllers/action_controller.py"
echo "   - scripts/triggers/face_trigger.py"
echo "   - scripts/triggers/ros_trigger.py"
echo "   - scripts/main.py"
echo ""
echo "2. Make sure ROS is sourced:"
echo "   source /opt/ros/noetic/setup.bash"
echo "   source ~/ros_ws/devel/setup.bash"
echo ""
echo "3. Run the system:"
echo "   cd $PROJECT_DIR/scripts"
echo "   python3 main.py"
echo ""
echo "=========================================="
