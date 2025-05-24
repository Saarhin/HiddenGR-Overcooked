import cv2
import os

# Set the directory containing the image files
#image_dir = 'misc/game/record/KirinTest1_agents2_seed1_model1-bd_model2-bd'
#image_dir = 'misc/game/record/KirinTest2_agents2_seed1_model1-bd_model2-bd'
#image_dir = 'misc/game/record/partial-divider_salad_agents2_seed1_model1-bd_model2-bd'
#image_dir = 'misc/game/record/KirinTest3_agents2_seed1_model1-bd_model2-bd'
#image_dir = 'misc/game/record/DesignOne_agents2_seed1_model1-bd_model2-bd'
#image_dir = 'misc/game/record/DesignOne_agents2_seed1'
#image_dir = 'misc/game/record/DesignTwo_agents2_seed1'
#image_dir = 'misc/game/record/DesignTwo_agents2_seed1_model1-bd_model2-bd'
#image_dir = 'misc/game/record/DesignThree_agents2_seed1'
#image_dir = 'misc/game/record/DesignThree_agents2_seed1_model1-bd_model2-bd'
#image_dir = 'misc/game/record/DesignFour_agents2_seed1'
#image_dir = 'misc/game/record/DesignFour_agents2_seed1_model1-bd_model2-bd'

image_dir = 'misc/game/record/SarahDesignOne_agents2_seed1'

# Get a list of all the image files in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
image_files.sort()

# Loop through the image files and display them
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    image = cv2.imread(image_path)
    
    # Display the image
    cv2.imshow('Image Sequence', image)
    
    # Wait for a key press (5000 ms = 5 seconds per image)
    # Press any key to move to the next image
    key = cv2.waitKey(5000)
    
    # Break the loop if 'q' is pressed
    if key == ord('q'):
        break

# Close all OpenCV windows
cv2.destroyAllWindows()