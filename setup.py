#!/usr/bin/env python3
"""
Setup script for University Dropout Prediction project
Handles environment setup, dependency installation, and data verification
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print("✅ Python version:", sys.version.split()[0])
    return True

def install_requirements():
    """Install project dependencies"""
    try:
        print("📦 Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def verify_data_structure():
    """Verify that required data files exist"""
    required_files = [
        "data/processed/df_objetivo/df_escalado.xlsx",
        "data/processed/df_objetivo/2024-2.xlsx"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  Missing data files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("Please ensure all data files are properly placed.")
        return False
    
    print("✅ Data structure verified")
    return True

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn',
        'sklearn', 'xgboost', 'imblearn', 'shap'
    ]
    
    failed_imports = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            failed_imports.append(package)
    
    if failed_imports:
        print("❌ Failed to import:", ", ".join(failed_imports))
        return False
    
    print("✅ All packages imported successfully")
    return True

def test_data_loading():
    """Test if main dataset can be loaded"""
    try:
        df = pd.read_excel("data/processed/df_objetivo/df_escalado.xlsx")
        print(f"✅ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Check for target variable
        if 'RIESGO_DESERCION' in df.columns:
            print(f"✅ Target variable found: {df['RIESGO_DESERCION'].value_counts().to_dict()}")
        else:
            print("⚠️  Target variable 'RIESGO_DESERCION' not found")
            
        return True
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False

def create_jupyter_config():
    """Create basic Jupyter configuration"""
    try:
        subprocess.check_call([sys.executable, "-m", "ipykernel", "install", "--user", "--name", "dropout-prediction"])
        print("✅ Jupyter kernel installed")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Could not install Jupyter kernel")
        return False

def main():
    """Main setup function"""
    print("🚀 University Dropout Prediction - Project Setup")
    print("=" * 50)
    
    # Change to project directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    checks = [
        ("Python Version", check_python_version),
        ("Install Requirements", install_requirements),
        ("Data Structure", verify_data_structure),
        ("Package Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Jupyter Config", create_jupyter_config)
    ]
    
    results = {}
    for check_name, check_func in checks:
        print(f"\n🔍 {check_name}...")
        results[check_name] = check_func()
    
    print("\n" + "=" * 50)
    print("📋 Setup Summary:")
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {check_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 Setup completed successfully!")
        print("\n📖 Next steps:")
        print("   1. Launch Jupyter: jupyter notebook")
        print("   2. Navigate to notebooks/ directory")
        print("   3. Start with 01_data_preparation.ipynb")
    else:
        print("\n⚠️  Setup completed with issues. Please resolve the failed checks above.")

if __name__ == "__main__":
    main()
