#!/usr/bin/env python3

import pandas as pd
import numpy as np


sub1 = pd.read_csv('b3_0.8793.csv')
sub2 = pd.read_csv('b3_0.8791.csv')
sub3 = pd.read_csv('b1_0.8790.csv')
sub4 = pd.read_csv('b1_0.8789.csv')
sub5 = pd.read_csv('ens_0.8790.csv')

predictions = 0.75*(sub1.target.values+sub2.target.values+sub3.target.values+sub4.target.values)/4.0+0.25*sub5.target.values


test = pd.read_csv('sample_submission.csv')

test['target'] = predictions
test[['id', 'target']].to_csv('submission.csv', index=False)