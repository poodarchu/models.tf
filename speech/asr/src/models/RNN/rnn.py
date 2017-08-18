import os 
import tensorflow as tf
from configparser import ConfigParser

from models.RNN.utils import variable_on_cpu

def SimpleLSTM(config_path, input_tensor, seq_length):
	parser = ConfigParser(os.environ)
	parser.read(config_path)

	# SimpleLSTM
	n_character = parser.getint('simplelstm', 'n_character')
	b1_stddev = parser.getfloat('simplelstm', 'b1_stddev')
	h1_stddev = parser.getfloat('simplelstm', 'h1_stddev')
	n_layers = parser.getint('simplelstm', 'n_layers')
	n_hidden_units = parser.getint('simplelstm', 'n_hidden_units')

	# Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
	# batch_x_shape = tf.shape(batch_x)

	input_tensor_shape = tf.shape(input_tensor)
	n_items = input_tensor_shape[0]

	with tf.name_scope('lstm'):
		W = tf.get_variable('W', 
							shape=[n_hidden_units, n_character], 
							# initializer=tf.truncated_normal_initializer(stddev=h1_stddev),
							initializer=tf.random_normal_initializer(stddev=h1_stddev))
		b = tf.get_variable('b',
							shape=[n_character],
							initializer=tf.random_normal_initializer(stddev=b1_stddev))

		# Define the cell
		# can be: tf.contrib.rnn.BasicRNNCell, tf.contrib.rnn.GRUCell
		cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, state_is_tuple=True)

		# Stacking rnn cells
		stack = tf.contrib.rnn.MultiRNNCell([cell]*n_layers, state_is_tuple=True)

		# Get layer activations (second output is the final state of the layer, do not needed)
		outputs, _ = tf.nn.dynamic_rnn(stack, input_tensor, seq_length, time_major=False, dtype=tf.float32)

		# Reshape to apply the same weights over the timesteps
		outputs = tf.reshape(outputs, [-1, n_hidden_units])

		# Perform affine transformation to layer output
		# Multiply by weights (linear transformation), add bias(translation)
		logits = tf.add(tf.matmul(outputs, W), b)

		tf.summary.histogram('weights', W)
		tf.summary.histogram('biases', b)
		tf.summary.histogram('activations', logits)

		# Reshaping back to original shape
		logits = tf.reshape(logits, [n_items, -1, n_character])

		# Put time as the major axis
		logits = tf.transpose(logits, (1, 0, 2))

		summary_op = tf.summary.merge_all()

	return logits, summary_op



def BiRNN(config_path, batch_x, seq_length, n_input, n_context):
	parser = ConfigParser(os.environ)
	parser.read(conf_path)

	dropout = [float(x) for x in parser.get('birnn', 'dropout_rates').split(',')]
	relu_clip=  parser.getint('birnn', 'relu_clip')

	b1_stddev = parser.getfloat('birnn', 'b1_stddev')
    h1_stddev = parser.getfloat('birnn', 'h1_stddev')
    b2_stddev = parser.getfloat('birnn', 'b2_stddev')
    h2_stddev = parser.getfloat('birnn', 'h2_stddev')
    b3_stddev = parser.getfloat('birnn', 'b3_stddev')
    h3_stddev = parser.getfloat('birnn', 'h3_stddev')
    b5_stddev = parser.getfloat('birnn', 'b5_stddev')
    h5_stddev = parser.getfloat('birnn', 'h5_stddev')
    b6_stddev = parser.getfloat('birnn', 'b6_stddev')
    h6_stddev = parser.getfloat('birnn', 'h6_stddev')

    n_hidden_1 = parser.getint('birnn', 'n_hidden_1')
    n_hidden_2 = parser.getint('birnn', 'n_hidden_2')
    n_hidden_5 = parser.getint('birnn', 'n_hidden_5')
    n_cell_dim = parser.getint('birnn', 'n_cell_dim')

    n_hidden_3 = int(eval(parser.get('birnn', 'n_hidden_3')))
    n_hidden_6 = parser.getint('birnn', 'n_hidden_6')

    # Input shape: [batch_size, n_steps, n_input + 2*n_input*n_context]
    batch_x_shape = tf.shape(batch_x)

    # Reshaping `batch_x` to a tensor with shape `[n_steps*batch_size, n_input + 2*n_input*n_context]`.
    # This is done to prepare the batch for input into the first layer which expects a tensor of rank `2`.

    # Permute n_steps and batch_size
    batch_x = tf.transpose(batch_x, [1, 0, 2])
    # Reshape to prepare input for first layer
    batch_x = tf.reshape(batch_x, [-1, n_input + 2 * n_input * n_context])

    # The next three blocks will pass 'batch_x' through three hidden layers with clipped RELU activations and dropout

    # 1st layer
    with tf.name_scope('fc1'):
    	b1 = variable_on_cpu('b1', [n_hidden_1], tf.random_normal_initializer(stddev=b1_stddev))
    	h1 = variable_on_cpu('h1', [n_input + 2 * n_input * n_context, n_hidden_1], tf.random_normal_initializer(stddev=h1_stddev))

    	layer_1 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(batch_x, h1), b1)), relu_clip)
    	layer_1 = tf.nn.dropout(layer_1, (1.0-dropout[0]))

    	tf.summary.histogram("weights", h1)
    	tf.summary.histogram("biases", b1)
    	tf.summary.histogram("activations", layer_1)

    # 2nd layer
    with tf.name_scope('fc2'):
        b2 = variable_on_cpu('b2', [n_hidden_2], tf.random_normal_initializer(stddev=b2_stddev))
        h2 = variable_on_cpu('h2', [n_hidden_1, n_hidden_2], tf.random_normal_initializer(stddev=h2_stddev))
        layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), relu_clip)
        layer_2 = tf.nn.dropout(layer_2, (1.0 - dropout[1]))

        tf.summary.histogram("weights", h2)
        tf.summary.histogram("biases", b2)
        tf.summary.histogram("activations", layer_2)


    # 3rd layer
    with tf.name_scope('fc3'):
        b3 = variable_on_cpu('b3', [n_hidden_3], tf.random_normal_initializer(stddev=b3_stddev))
        h3 = variable_on_cpu('h3', [n_hidden_2, n_hidden_3], tf.random_normal_initializer(stddev=h3_stddev))
        layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), relu_clip)
        layer_3 = tf.nn.dropout(layer_3, (1.0 - dropout[2]))

        tf.summary.histogram("weights", h3)
        tf.summary.histogram("biases", b3)
        tf.summary.histogram("activations", layer_3)

    # Create the forward and backward LSTM units.
    # Inputs have length 'n_cell_dim'.
    # LSTM forget gate bias initialized at 1.0(default), meaning less forgeting at the beginning of training.
    with tf.name_scope('lstm'):
    	# Forward direction cell:
        # n_cell_dim is num_units, it's the size of the LSTM's hidden state
    	lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
    	lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, input_keep_prob=1.0-dropout[3], output_keep_prob=1.0=dropout[3],
    													# seed = random_seed,
    													)
    	# Backward direction cell
    	lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_cell_dim, forget_bias=1.0, state_is_tuple=True)
    	lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, input_keep_prob=1.0-dropout[4], output_keep_prob=1.0-dropout[4])

    	# layer_3 is now reshaped into [n_steps, batch_size, 2*n_cell_dim]
    	# as the LSTM BRNN expects its input to be of shape [max_time, batch_size, input_size]
    	layer_3 = tf.reshape(layer_3, [-1, batch_x_shape[0], n_hidden_3])

    	# now we feed layer_3 into the LSTM BRNN cell and obtain the LSTM BRNN output
    	outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell, inputs=layer_3, dtype=tf.float32, time_major=True, sequence_length=seq_length)

    	tf.summary.histogram("activations", outputs)

    	outputs = tf.concat(outputs, 2)
    	outputs = tf.reshape(outputs, [-1, 2*n_cell_dim])

    with tf.name_scope('fc5'):
    	b5 = variable_on_cpu('b5', [n_hidden_5], tf.random_normal_initializer(stddev=b5_stddev))
    	h5 = variable_on_cpu('h5', [(2*n_cell_dim), n_hidden_5], tf.random_normal_initializer(stddev=h5_stddev))
    	layer_5 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(outputs, h5), b5)), relu_clip)
        layer_5 = tf.nn.dropout(layer_5, (1.0 - dropout[5]))

        tf.summary.histogram("weights", h5)
        tf.summary.histogram("biases", b5)
        tf.summary.histogram("activations", layer_5)

    # now apply the weight matrix h6 and bias b6 to the output of layer_5
    # creating n_classes dimensional vectors, the logits.
    with tf.name_scope('fc6'):
    	b6 = variable_on_cpu('b6', [n_hidden_6], tf.random_normal_initializer(stddev=b6_stddev))
    	h6 = variable_on_cpu('h6', [n_hidden_5, n_hidden_6], tf.random_normal_initializer(stddev=h6_stddev))
    	layer_6 = tf.add(tf.matmul(layer_5, h6), b6)

    	tf.summary.histogram('weights', h6)
    	tf.summary.histogram('biases', b6)
    	tf.summary.histogram('activations', layer_6)
    # Finally reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6] 
    # to the slightly more useful shape [n_step, batch_size, n_hidden_6]
    # Note, that this differs from the input in that it is time-major
    layer_6 = tf.reshape(layer_6, [-1, batch_x_shape[0], n_hidden_6])

    summary_op = tf.summary.merge_all()

    # Output shape: [n_steps, batch_size, n_hidden_6]
    return layer_6, summary_op

	