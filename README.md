# Description

Bachelor's thesis by [Björn Mattsson](https://www.linkedin.com/in/björn-mattsson-02357b70) and 
[Olof Steinert](https://www.linkedin.com/in/olof-steinert/) in Economics at 
[University of Gothenburg](http://handels.gu.se/).

An attempt ot use ML techniques to predict bankruptcies in companies.

# Instructions

* Unzip files from https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data# into data/
* run `python create_csv_files.py`
* To run experiment: define it in `config.yml`, run experiment with `python run.py config.yml`. Results (and config file for future reference) will be saved in `output/TIMESTAMP`.

## Version info

* Python 3.5.2
* Tensorflow 1.1.0rc1
