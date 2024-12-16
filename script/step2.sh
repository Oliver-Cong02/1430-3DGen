# 3DGen: Step II
# Step 1: Set up the required environment for ControlNet-v1-1

mv script/ControlNet-v1-1/depth.py extern/ControlNet-v1-1/depth.py
mv script/ControlNet-v1-1/gradio_depth.py extern/ControlNet-v1-1/gradio_depth.py
# Step 2: Prepare the inputs (depth map) from extern/GeoWizard/geowizard/output_object
cd extern/ControlNet-v1-1
python depth.py # Please follow the instructions in the terminal locally

# Or you can run online gradio interface like below
python gradio_depth.py

# Step 3: Check the output depth map and npy file in extern/ControlNet-v1-1/output