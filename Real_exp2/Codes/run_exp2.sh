rm ../Real_Data_Results/*.pkl 
python generate_human_error.py
python train.py
python test.py
