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

