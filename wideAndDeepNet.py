from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, concatenate


class WideAndDeepModel(Model):
    def __init__(self, units=15, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = Dense(units, activation=activation)
        self.hidden2 = Dense(units, activation=activation)
        self.main_out = Dense(1)
        self.aux_out = Dense(1)

    def call(self, inputs):
        input_1, input_2 = inputs
        hidden1 =  self.hidden1(input_2)
        hidden2 =  self.hidden2(hidden1)

        concat = concatenate([input_1, hidden2])
        main_out = self.main_out(concat)
        aux_out = self.aux_out(main_out)
        return main_out, aux_out
    
model = WideAndDeepModel()
