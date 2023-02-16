#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Antonio Jesús Banegas-Luna"
__version__ = "1.0"
__maintainer__ = "Antonio"
__email__ = "ajbanegas@ucam.edu"
__status__ = "Production"

import pandas as pd
from ConsensusBase import ConsensusBase

class ConsensusAverageMean(ConsensusBase):

    def __init__(self, folder):
        super(ConsensusAverageMean, self).__init__(folder)
        self.title = 'Average mean'

    def consensus(self):
        print("Computing average mean")

        df = pd.concat([self.df_g, self.df_l], ignore_index=True)
        df = df[[self.FEATURE, self.ATTR]]

        # average mean of the attributions
        df_mean = df.groupby([self.FEATURE])[self.ATTR].mean().to_frame().reset_index()
        
        # output
        #features = df_mean[self.FEATURE].to_numpy()
        #attrs = df_mean[self.ATTR].to_numpy()
        #return features, attrs
        return df_mean
