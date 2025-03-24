# Inception Score MindSpore

This code is a mindspore implementation of Occupational Biases in Norwegian and Multilingual Language Models which is available at https://github.com/samiatouileb/biases-norwegian-multilingual-lms paperswidthcocde link is https://paperswithcode.com/paper/occupational-biases-in-norwegian-and.

# Usage

To run the code: 

```
python ./codes/compute_scores.py --task occupation --template_file ./templates/templates_er.txt --lm_model mbert --output_file test_NorBERT.csv
```

Where:

- "--task" is of ``type=str'', and can only be "occupation" for now.
- "--template_file", is of "type=str", and should provide the path to the template file.
- "--lm_model", is of "type=str",  and referes to the pretrained language models to use (options: "norbert", "nbbert", "mbert", "roberta", "nbbertLarge", "nbroberta", "norbert2").
- "--output_file", is of "type=str", and should provide the path to output file with sentence scores")