# 3DGen: Step I
# Step 1: Set up the required environment for GeoWizard

mv script/Geowizard/run.sh extern/GeoWizard/geowizard/run.sh
# Step 2: Prepare the inputs in extern/GeoWizard/geowizard/input/example_object
cd extern/GeoWizard/geowizard
sh run.sh

# Step 3: Check the output depth map and npy file in extern/GeoWizard/geowizard/output_object