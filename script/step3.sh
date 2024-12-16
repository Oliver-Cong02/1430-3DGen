# 3DGen: Step III
# Step 1: Set up the required environment for InstantMesh

# Step 2: Prepare the modified image prompt from extern/ControlNet-v1-1/output
cd extern/InstantMesh
python app.py # which will open a public gradio interface where you can operate.

# Or you can run locally like below
python run.py configs/instant-mesh-large.yaml "image_path" --save_video

# Step 3: Remove the texture and check the generated 3D assets