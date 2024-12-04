import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

class ZoneoutRNN(nn.Cell):
    def __init__(self,forward_cell,backward_cell,zoneout_prob,bidrectional=True,dropout_rate=0.5):
        super().__init__()
        self.forward_cell=forward_cell
        self.backward_cell=backward_cell
        self.zoneout_prob=zoneout_prob
        self.bidrectional=bidrectional
        self.dropout_rate=dropout_rate

        if self.bidrectional:
            if self.forward_cell.hidden_size!= self.backward_cell.hidden_size:
                raise TypeError("The forward cell should be the same as backward!")
        if isinstance(forward_cell,nn.LSTM):
            if not isinstance(zoneout_prob,tuple):
                raise TypeError("The LSTM zoneout_prob must be a tuple!")
        elif isinstance(forward_cell,nn.GRU):
                raise TypeError("The GRU zoneout_ptob must be a float number!")
        elif isinstance(forward_cell,nn.RNN):
                raise TypeError("The RNN zoneout_prob must be a float number!")
    
    @property
    def hidden_size(self):
        return self.forward_cell.hidden_size()
    @property
    def input_size(self):
        return self.forward_cell.input_size()
    
    def construct(self,forward_input,backward_input,forward_state,backward_state):
        if self.bidrectional==True:
            forward_new_state =self.forward_cell(forward_input,forward_state)
            backward_new_state=self.backward_cell(backward_input,backward_state)
            if isinstance(self.forward_cell,nn.LSTMCell):
                forward_h,forward_c=forward_state
                forward_new_h,forward_new_c=forward_new_state

                backward_h,backward_c=backward_state
                backward_new_h,backward_new_c=backward_new_state
                zoneout_prob_c,zoneout_prob_h=self.zoneout_prob

                forward_new_h=(1-zoneout_prob_h)*ops.dropout(forward_new_h,p=self.dropout_rate,training=self.training)+forward_h
                forward_new_c=(1-zoneout_prob_c)*ops.dropout(forward_new_c,p=self.dropout_rate,training=self.training)+forward_c

                backward_new_h = (1 - zoneout_prob_h) * ops.dropout(backward_new_h, p=self.dropout_rate,
                                                                 training=self.training) + backward_h
                backward_new_c = (1 - zoneout_prob_c) * ops.dropout(backward_new_c, p=self.dropout_rate,
                                                                 training=self.training) + backward_c

                forward_new_state=(forward_new_h,forward_new_c)
                backward_new_state=(forward_new_h,backward_new_c)
                forward_output=forward_new_h
                backward_output=backward_new_h
            
            else:
                forward_h = forward_state
                forward_new_h = forward_new_state

                backward_h = backward_state
                backward_new_h = backward_new_state
                zoneout_prob_h = self.zoneout_prob

                forward_new_h = (1 - zoneout_prob_h) * ops.dropout(forward_new_h, p=self.dropout_rate,
                                                                 training=self.training) + forward_h
                backward_new_h = (1 - zoneout_prob_h) * ops.dropout(backward_new_h, p=self.dropout_rate,
                                                                  training=self.training) + backward_h

                forward_new_state = forward_new_h
                backward_new_state = backward_new_h
                forward_output = forward_new_h
                backward_output = backward_new_h

            
            return forward_output, backward_output, forward_new_state, backward_new_state
        else:
            forward_new_state=self.forward_cell(forward_input,forward_state)
            if isinstance(self.forward_cell,nn.LSTMCell):
                forward_h, forward_c = forward_state
                forward_new_h, forward_new_c = forward_new_state

                zoneout_prob_c, zoneout_prob_h = self.zoneout_prob

                forward_new_h = (1 - zoneout_prob_h) * ops.dropout(forward_new_h, p=self.dropout_rate,
                                                                 training=self.training) + forward_h
                forward_new_c = (1 - zoneout_prob_c) * ops.dropout(forward_new_c, p=self.dropout_rate,
                                                                 training=self.training) + forward_c
                forward_new_state = (forward_new_h, forward_new_c)
                forward_output = forward_new_h

            else:
                forward_h = forward_state
                forward_new_h = forward_new_state

                zoneout_prob_h = self.zoneout_prob

                forward_new_h = (1 - zoneout_prob_h) * ops.dropout(forward_new_h, p=self.dropout_rate,
                                                                 training=self.training) + forward_h

                forward_new_state = forward_new_h
                forward_output = forward_new_h
            return forward_output, forward_new_state


