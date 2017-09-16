# NuisanceRegression
Performs smoothing/regfilt/nuisance regression after FMRIPREP

## Installation
Partially tested: works on python 2.7 and 3.5
Python Dependencies:
- niworkflows
- pybids

```pip install niworkflows pybids```

OS Dependencies:
- graphviz

UBUNTU:
```sudo apt install graphviz```

MAC OSX (using homebrew)
```brew install graphviz```

## Usage
```usage: preprocess.py [-h]
                     [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]]
                     [-w WORK_DIR] [-sm SMOOTH] [-l LOW_PASS] [-f]
                     [-c CONFOUNDS] [-t TASK_ID] [-sp SPACE]
                     [--variant VARIANT] [--exclude_variant] [-r RES]
                     [--run RUN] [--ses SES] [--graph]
                     deriv_pipe_dir output_dir

NuisanceRegression BIDS arguments

positional arguments:
  deriv_pipe_dir        FMRIPREP directory
  output_dir            output directory

optional arguments:
  -h, --help            show this help message and exit
  --participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]
                        The label(s) of the participant(s) that should be
                        analyzed. The label corresponds to
                        sub-<participant_label> from the BIDS spec (so it does
                        not include "sub-"). If this parameter is not provided
                        all subjects should be analyzed. Multiple participants
                        can be specified with a space separated list.
  -w WORK_DIR, --work_dir WORK_DIR
                        Directory where all intermediate files are stored

Options for preprocessing:
  -sm SMOOTH, --smooth SMOOTH
                        select a smoothing kernel (mm)
  -l LOW_PASS, --low_pass LOW_PASS
                        low pass filter
  -f, --regfilt         Do non-aggressive filtering from ICA-AROMA
  -c CONFOUNDS, --confounds CONFOUNDS
                        The confound column names that are to be included in
                        nuisance regression

Options for selecting images:
  -t TASK_ID, --task_id TASK_ID
                        select a specific task to be processed
  -sp SPACE, --space SPACE
                        select a bold derivative in a specific space to be
                        used
  --variant VARIANT     select a variant bold to process
  --exclude_variant     exclude the variant from FMRIPREP
  -r RES, --res RES     select a resolution to analyze
  --run RUN             select a run to analyze
  --ses SES             select a session to analyze

Options for miscellaneous abilities:
  --graph               generates a graph png of the workflow```
  
Example Call:
```./preprocess.py ~/devel/tmp/sample_bids/derivatives/fmriprep ~/devel/tmp/sample_bids/derivatives -w ~/devel/tmp/work -sm 6 -l 0.1 --exclude_variant```

