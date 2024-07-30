# Lester


taskset --cpu-list 0-3 python experiment__retraining_time.py --num_customers 10 --num_repetitions 7 |tee retraining10.txt
taskset --cpu-list 0-3 python experiment__retraining_time.py --num_customers 100 --num_repetitions 7 |tee retraining100.txt
taskset --cpu-list 0-3 python experiment__retraining_time.py --num_customers 1000 --num_repetitions 7 |tee retraining1000.txt