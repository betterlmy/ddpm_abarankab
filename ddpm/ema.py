class EMA():
    def __init__(self, decay):
        # decay
        self.decay = decay
        
    
    def update_average(self, old, new):
        # 每次更新模型参数后,EMA会根据一个超参数α,更新最终跟踪的值:
        # 由于刚开始训练不稳定，得到的梯度给更小的权值更为合理，所以EMA会有效。
        if old is None:
            return new
        return old * self.decay + (1 - self.decay) * new

    def update_model_average(self, ema_model, current_model):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old, new = ema_params.data, current_params.data
            ema_params.data = self.update_average(old, new)