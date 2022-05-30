Install a few of extra requirements:
```bash
python3 -m pip install -r requirements.txt
```

## Run

To run all benchmarks:
```bash
python3 run.py
```

## Result

The benchmark results will be stored in the `results` folder in your current directory.
If you wish to save the results as a single json file (`./results/results.json`):
```bash
python3 deserialize.py
```
Or you can specify the input and output path:
```bash
python3 deserialize.py --folder PATH_OF_RESULTS_FOLDER --output_path PATH_YOU_WIHS_TO_STORE
```

## Tools

After getting benchmark results (`./results`), you can use a visualization tool to profile performance problems:
```bash
python3 visualization.py
```

You can specify the results file path:
```bash
python3 visualization.py --folder PATH_OF_RESULTS_FOLDER
```

The default host and port is `localhost:5006\visualization`.
If you want to enable remote access, take the following steps:
```bash
python3 visualization.py --host YOUR_IP_ADDRESS --port PORT_YOU_WISH_TO_USE
```
