python -u train.py -s data/ref/refnerf/ball --eval --iterations 61000 --white_background
python -u train.py -s data/ref/refnerf/car --eval --iterations 61000 --white_background
# python -u train.py -s data/ref/refnerf/coffee --eval --iterations 61000 --white_background
python -u train.py -s data/ref/refnerf/helmet --eval --iterations 61000 --white_background
python -u train.py -s data/ref/refnerf/teapot --eval --iterations 61000 --white_background
python -u train.py -s data/ref/refnerf/toaster --eval --iterations 61000 --white_background --longer_prop_iter 24_000