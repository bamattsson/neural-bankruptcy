# Description

[Bachelor's thesis](https://gupea.ub.gu.se/handle/2077/54283) by [Björn Mattsson](https://www.linkedin.com/in/björn-mattsson-02357b70) and 
[Olof Steinert](https://www.linkedin.com/in/olof-steinert/) in Economics at 
[University of Gothenburg](http://handels.gu.se/).

## Abstract
Estimating the risk of corporate bankruptcies is of large importance to creditors and investors. For this reason bankruptcy prediction constitutes an important area of research. In recent years artificial intelligence and machine learning methods have achieved promising results in corporate bankruptcy prediction settings. Therefore, in this study, three machine learning algorithms, namely random forest, gradient boosting and an artificial neural network were used to predict corporate bankruptcies. Polish companies between 2000 and 2013 were studied and the predictions were based on 64 different financial ratios.

The obtained results are in line with previously published findings. It is shown that a very good predictive performance can be achieved with the machine learning models. The reason for the impressive predictive performance is analysed and it is found that the missing values in the data set play an important role. It is observed that prediction models with surprisingly good performance could be achieved from only information about the missing values of the data and with the financial information excluded.

Keywords: Economics, Corporate bankruptcy prediction, Machine learning, Neural networks, Missing values


# How to run

* Unzip files from https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data# into data/
* run `python create_csv_files.py`
* To run experiment: define it in `config.yml`, run experiment with `python run.py config.yml`. Example files can be found in `configs/`
* Results (and config file for future reference) will be saved in `output/TIMESTAMP`.

## Version info

* Python 3.5.2
* Tensorflow 1.1.0rc1
