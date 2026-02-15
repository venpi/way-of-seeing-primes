# Data

The full 68M harvest file (harvest_68M_v3_1.json, ~7GB) is not included 
due to size. To regenerate it:

python scripts/prime_harvest_v3_1.py --intervals 68000000 --output harvest_68M_v3_1.json --verbose

This takes ~2-3 hours and requires ~8GB RAM.

For quick testing, generate a smaller dataset:

python scripts/prime_harvest_v3_1.py --intervals 100000 --output harvest_100k.json --verbose