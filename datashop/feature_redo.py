class NumericFeature(Feature):
    self.feature_type = 'numeric'
    self.report['Feature_Type'] = 'Numeric'
    self.report['Mean'] = self.series.mean()
    self.report['Median'] = self.series.median()
    
    self.plot_map = {
        'figure':{
                'figsize':[12,12],
                'grid_size':[12,12]
                },
        'plots':[
            {
            'type':self.box_plot,
            'data':self.series,
            'location':[slice(None,2),slice(None)]
            },
            {
            'type':self.histo,
            'data':self.series,
            'location':[slice(2,9),slice(None)]
            }
            ]
    }

def StringFeature(self):

        self.feature_type = 'string'
        self.report['Feature_Type'] = 'String'
        self.str_lengths = self.series.fillna(' ').apply(len)
        
        self.report['Mean Length'] = self.str_lengths.mean()
        self.report['Median Length'] = self.str_lengths.median()
        
        self.series.fillna(' ',inplace = True)
        self.chart_words = 25
        self.countvec = CountVectorizer(stop_words='english')
        self.wordcounts= self.countvec.fit_transform(self.series)
        self.word_totals = self.wordcounts.toarray().sum(axis=0)
        self.words = self.countvec.get_feature_names()
        self.count_dict = dict(zip(self.words,self.word_totals))
        self.count_series = pd.Series(self.count_dict)
        self.count_series.sort_values(ascending = False,inplace=True)
        
        self.plot_map = {
            'figure':{
                    'figsize':[16,12],
                    'grid_size':[12,12]
                    },
            'plots':[
                    {
                    'type':self.box_plot,
                    'data':self.str_lengths,
                    'location':[slice(None,2),slice(None)]
                    },
                    {
                    'type':self.histo,
                    'data':self.str_lengths,
                    'location':[slice(2,9),slice(None)]
                    },
                    {
                    'type':self.bar_plot,
                    'data':self.count_series.head(self.chart_words),
                    'location':[slice(9,None),slice(None)]
                    }
                    ]
                }

def DateFeature(self):
        
        self.feature_type = 'date'

        self.series.sort_index(ascending=True,inplace=True)
        
        self.report["Earliest Point"] = self.series.index[0]
        self.report["Latest Point"] = self.series.index[-1]
        
        self.series.sort_index(ascending=True,inplace=True)
        
        self.plot_map = {
            'figure':{
                        'figsize':[16,6],
                        'grid_size':[16,6]
                        },
            'plots':[
                {
                'type':self.line_plot,
                'data':self.series,
                'location':[slice(None),slice(None)]
                }
                ]
            }
def Categorical(self):

        self.feature_type = 'categorical'
        
        display(self.report)
        
        self.series = self.series.astype('category')
        
        self.plot_map = {
            'figure':{
                        'figsize':[12,.75*len(self.val_counts)],
                        'grid_size':[12,12]
                        },
            'plots':[
                {
                'type':self.box_plot,
                'data':self.series,
                'location':[slice(0,2),slice(None)]
                }
                ]
            }