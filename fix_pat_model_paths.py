#!/usr/bin/env python3
"""Fix PAT model path configuration for Docker deployment."""

import os
import json
from pathlib import Path

# The actual model names as they appear in S3 and in the error logs
ACTUAL_MODEL_MAPPING = {
    "small": "PAT-S_29k_weight_transformer.h5",  
    "medium": "PAT-M_29k_weight_transformer.h5",  
    "large": "PAT-L_91k_weight_transformer.h5"   
}

# Expected names in our code
EXPECTED_MODEL_MAPPING = {
    "small": "PAT-S_29k_weights.h5",
    "medium": "PAT-M_29k_weights.h5", 
    "large": "PAT-L_29k_weights.h5"
}

def update_pat_service():
    """Update pat_service.py to use correct model paths and names."""
    pat_service_path = Path("src/clarity/ml/pat_service.py")
    
    with open(pat_service_path, 'r') as f:
        content = f.read()
    
    # Update the model names to match what's actually in S3
    content = content.replace(
        '"PAT-S_29k_weights.h5"',
        '"PAT-S_29k_weight_transformer.h5"'
    )
    content = content.replace(
        '"PAT-M_29k_weights.h5"',
        '"PAT-M_29k_weight_transformer.h5"'
    )
    content = content.replace(
        '"PAT-L_29k_weights.h5"',
        '"PAT-L_91k_weight_transformer.h5"'
    )
    
    with open(pat_service_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated {pat_service_path}")

def update_download_script():
    """Update download_models.sh to download the correct model files."""
    script_path = Path("scripts/download_models.sh")
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Update model paths to match actual S3 files
    content = content.replace(
        'MODELS["PAT-S"]="${PAT_S_MODEL_PATH:-${S3_BUCKET}/pat/PAT-S_29k_weights.h5}"',
        'MODELS["PAT-S"]="${PAT_S_MODEL_PATH:-${S3_BUCKET}/pat/PAT-S_29k_weight_transformer.h5}"'
    )
    content = content.replace(
        'MODELS["PAT-M"]="${PAT_M_MODEL_PATH:-${S3_BUCKET}/pat/PAT-M_29k_weights.h5}"',
        'MODELS["PAT-M"]="${PAT_M_MODEL_PATH:-${S3_BUCKET}/pat/PAT-M_29k_weight_transformer.h5}"'
    )
    content = content.replace(
        'MODELS["PAT-L"]="${PAT_L_MODEL_PATH:-${S3_BUCKET}/pat/PAT-L_29k_weights.h5}"',
        'MODELS["PAT-L"]="${PAT_L_MODEL_PATH:-${S3_BUCKET}/pat/PAT-L_91k_weight_transformer.h5}"'
    )
    
    with open(script_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated {script_path}")

def update_ecs_task_definition():
    """Update ECS task definition with correct model paths."""
    task_def_path = Path("ops/deployment/ecs-task-definition.json")
    
    with open(task_def_path, 'r') as f:
        task_def = json.load(f)
    
    # Find and update environment variables
    for container in task_def.get('containerDefinitions', []):
        if container.get('name') == 'clarity-backend':
            env_vars = container.get('environment', [])
            for var in env_vars:
                if var['name'] == 'PAT_S_MODEL_PATH':
                    var['value'] = "s3://clarity-ml-models-124355672559/pat/PAT-S_29k_weight_transformer.h5"
                elif var['name'] == 'PAT_M_MODEL_PATH':
                    var['value'] = "s3://clarity-ml-models-124355672559/pat/PAT-M_29k_weight_transformer.h5"
                elif var['name'] == 'PAT_L_MODEL_PATH':
                    var['value'] = "s3://clarity-ml-models-124355672559/pat/PAT-L_91k_weight_transformer.h5"
    
    with open(task_def_path, 'w') as f:
        json.dump(task_def, f, indent=2)
    
    print(f"✅ Updated {task_def_path}")

def create_symlink_script():
    """Create a script to create symlinks for backward compatibility."""
    symlink_script = """#!/bin/bash
# Create symlinks for backward compatibility with old model names
set -e

MODEL_DIR="${MODEL_DIR:-/app/models/pat}"
cd "$MODEL_DIR"

# Create symlinks if actual files exist
if [ -f "PAT-S_29k_weight_transformer.h5" ]; then
    ln -sf "PAT-S_29k_weight_transformer.h5" "PAT-S_29k_weights.h5"
    echo "✅ Created symlink for PAT-S model"
fi

if [ -f "PAT-M_29k_weight_transformer.h5" ]; then
    ln -sf "PAT-M_29k_weight_transformer.h5" "PAT-M_29k_weights.h5"
    echo "✅ Created symlink for PAT-M model"
fi

if [ -f "PAT-L_91k_weight_transformer.h5" ]; then
    ln -sf "PAT-L_91k_weight_transformer.h5" "PAT-L_29k_weights.h5"
    echo "✅ Created symlink for PAT-L model"
fi

# Also handle the /models/pat directory if it exists
if [ -d "/models/pat" ]; then
    cd /models/pat
    
    if [ -f "PAT-S_29k_weight_transformer.h5" ]; then
        ln -sf "PAT-S_29k_weight_transformer.h5" "PAT-S_29k_weights.h5"
    fi
    
    if [ -f "PAT-M_29k_weight_transformer.h5" ]; then
        ln -sf "PAT-M_29k_weight_transformer.h5" "PAT-M_29k_weights.h5"
    fi
    
    if [ -f "PAT-L_91k_weight_transformer.h5" ]; then
        ln -sf "PAT-L_91k_weight_transformer.h5" "PAT-L_29k_weights.h5"
    fi
fi
"""
    
    script_path = Path("scripts/create_model_symlinks.sh")
    with open(script_path, 'w') as f:
        f.write(symlink_script)
    
    # Make it executable
    os.chmod(script_path, 0o755)
    
    print(f"✅ Created {script_path}")

if __name__ == "__main__":
    print("🔧 Fixing PAT model path configuration...")
    
    # Update all the files
    update_pat_service()
    update_download_script()
    update_ecs_task_definition()
    create_symlink_script()
    
    print("\n✅ All files updated!")
    print("\n📝 Next steps:")
    print("1. Commit these changes")
    print("2. Build and push new Docker image")
    print("3. Deploy updated ECS task definition")
    print("4. The entrypoint script will create symlinks automatically")