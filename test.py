# tesing codes 1 it can save notebook and research.ipynb file
import os 

path="notebooks/research.ipynb"

dir,file=os.path.split(path)

os.makedirs(dir)

with open(path,"w") as f:
    pass





#tesing codes 2 it show log inforamtion in direct in terminal 


# Configure logging
#logging.basicConfig(
    #filename='virat.log',
    #level=logging.INFO,
    #format='%(asctime)s - %(levelname)s - %(message)s'
#)

# Console handler for logging to the console
#console_handler = logging.StreamHandler()
#console_handler.setLevel(logging.INFO)
#console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
#logging.getLogger().addHandler(console_handler)