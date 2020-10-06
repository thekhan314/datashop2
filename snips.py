      self.cull_series = pd.Series()

        self.base = int(base_threshold * 100)
    
        for x in range (1,self.base,1):
            
            self.current = round(float(x/100),2)
            self.cull_outliers(threshold = self.current)
            self.cull_loss = (len(self.series) -len(self.culled))/len(self.series)
            self.cull_series[str(self.current)] = self.cull_loss
        
        self.cull_fig,self.cull_ax = plt.subplots(figsize=(16,6))
        self.cull_ax.bar(self.cull_series.index,self.cull_series)
        self.cull_ax.tick_params(
                    axis='x', 
                    labelrotation=55,
                    length=10,
                    labelsize= 10
                    )