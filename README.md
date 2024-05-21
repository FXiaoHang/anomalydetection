step 1
cd code/
python process_data.py --session

step 2
cd code/
python main_sess.py --lp --full --n-epoch 60  && python inference_sess.py --lp 
