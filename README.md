# Introduction

Four step model for Singapore.

24/08/27 first upload: ICRS2024 

25/02/13 update: Disaggregated approach  

# Repository Structure

```
├── notebooks                
│   ├── 1.0-base-run: For Year 0
│   ├── 1.1-staged-policy: Reads Year 0 output, before running staged policy implementation
│   ├── 1.2-urgent-policy: Read Year 0 output before running urgent policy implementation
│   ├── 2.0-policy-development: Creates "policy" graphs for entry into four step model
│   ├── 2.1-subzone-nodes-mapping: Seeks and selects closest network node to subzone centroid as subzone location point
│   ├── 3.0-outputs-processing: Visualizes multiple runs model outputs.
├── Word file
│   ├── FSM Documention.docx: Description of the four step model and experimental setup
├── .py scripts
│   ├── trip_generation.py: step 1
│   ├── trip_distribution.py: step 2
│   ├── mode_split.py: step 3. Contains function for car ownership response.
│   ├── trip_assignment.py: step 4
│   ├── network_processing.py: contains functions for pre-processing networks
│   ├── outputs_processing.py: contains functions for processing outputs
│   ├── parallel_abrupt.py: script to execute parallel runs in terminal
├── data
│   ├── inputs
│   ├── outputs
```



