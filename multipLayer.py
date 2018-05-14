import tensorflow as tf 
import keras.backend as K 

class MultipLayer(Layer):

    #there are some difficulties for different types of shapes   
    #let's use a 'repeat_count' instead, increasing only one dimension
    def __init__(self, repeat_count,**kwargs):
        self.repeat_count = repeat_count
        super(MultipLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        #first, let's get the output_shape
        output_shape = self.compute_output_shape(input_shape)
        weight_shape = (1,) + output_shape[1:] #replace the batch size by 1


        self.kernel = self.add_weight(name='kernel',
                                      shape=weight_shape,
                                      initializer='ones',
                                      trainable=True)


        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    #here, we need to repeat the elements before multiplying
    def call(self, x):

        if self.repeat_count > 1:

             #we add the extra dimension:
             x = K.expand_dims(x, axis=1)

             #we replicate the elements
             x = K.repeat_elements(x, rep=self.repeat_count, axis=1)


        #multiply
        return x * self.kernel


    #make sure we comput the ouptut shape according to what we did in "call"
    def compute_output_shape(self, input_shape):

        if self.repeat_count > 1:
            return (input_shape[0],self.repeat_count) + input_shape[1:]
        else:
            return input_shape