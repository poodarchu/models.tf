import numpy as np
import unicodedata
import codecs
import re

import tensorflow as tf

# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1 # 0 is reserved to space

def normalize_txt_file(txt_tile, remove_apostrophe=True):
	with codecs.open(txt_file, encoding='utf-8') as open_txt_file:
		return normalize_text(open_txt_file.read(), remove_apostrophe=remove_apostrophe)


def normalized_text(original, remove_apostrophe=True):
	result = unicodedata.normalize("NFKD", original).encode("ascii", "ignore").decode()
    if remove_apostrophe:
    	result = result.replace("'", "")
    return re.sub("[^a-zA-Z']+", ' ', result).strip().lower()

def text_to_char_array(original):
	result = original.replace(' ', '  ')
	result = result.split(' ')

	result = np.hstack([SPACE_TOKEN if xt == '' else list(xt) for xt in result])

	return np.asarray([SPACE_TOKEN if xt == SPACE_TOKEN else ord(xt) - FIRST_INDEX for xt in result])

# Create a sparse representation of "sequences".
# sequences: a list of lists of type dtype where each element is a sequence
def sparse_tuple_from(sequences, dtype=np.int32):
	indices = []
	values = []
	for n, seq in enumerate(sequences):
		indices.extend(zip([n] * len(seq), range(len(seq))))
		values.extend(seq)
	indices = np.asarray(indices, dtype=np.int64)
	values = np.asarray(values, dtype=dtype)
	shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int64)

	return indices, values, shape

def sparse_tensor_value_to_texts(value):
	return sparse_tuple_to_texts((value.indices, value.values, value.dense_shape))

def sparse_tuple_to_texts(tuple):
	indices = tuple[0]
	values = tuple[1]
	results = [''] * tuple[2][0]
	for i in range(len(indices)):
		index = indices[i][0]
		c = values[i]
		c = ' ' if c == SPACE_INDEX else chr(c+FIRST_INDEX)
		result[index] = results[index] + c
	return results

def ndarray_to_text(value):
	results = ''
    for i in range(len(value)):
        results += chr(value[i] + FIRST_INDEX)
    return results.replace('`', ' ')

def gather_nd(params, indices, shape):
    rank = len(shape)
    flat_params = tf.reshape(params, [-1])
    multipliers = [reduce(lambda x, y: x * y, shape[i + 1:], 1) for i in range(0, rank)]
    indices_unpacked = tf.unstack(tf.transpose(indices, [rank - 1] + range(0, rank - 1)))
    flat_indices = sum([a * b for a, b in zip(multipliers, indices_unpacked)])
    return tf.gather(flat_params, flat_indices)

# The CTC implementation in TensorFlow needs labels in a sparse representation,
# but sparse data and queues don't mix well, so we store padded tensors in the
# queue and convert to a sparse representation after dequeuing a batch.
def ctc_label_dense_to_sparse(labels, label_lengths, batch_size):
	# The second dimension of labels must be equal to the longest label length in the batch
	correct_shape_assert = tf.assert_equal(tf.shape(labels)[1], tf.reduce_max(label_lengths))
	with tf.control_dependencies([correct_shape_assert]):
		labels = tf.identity(labels)

	label_shape = tf.shape(labels)
	num_batches_tns = tf.stack([label_shape[0]])
	max_num_labels_tns = tf.stack([label_shape[1]])

	def range_less_than(previous_state, current_input):
		return tf.expand_dims(tf.range(label_shape[1]), 0) < current_input

	init = tf.cast(tf.fill(max_num_labels_tns, 0), tf.bool)
	init = tf.expand_dims(init, 0)
	dense_mask = tf.scan(range_less_than, label_length, initializer=init, parallel_iterations=1)
	dense_mask = dense_mask[:, 0, :]

	label_array = tf.reshape(tf.tile(tf.range(0, label_shape[1]), num_batches_tns), label_shape)

	label_ind = tf.boolean_mask(label_array, dense_mask)

	batch_array = tf.transpose(tf.reshape(tf.tile(tf.range(0, label_shape[0]), max_num_labels_tns), tf.reverse(label_shape, [0])))

	batch_ind = tf.boolean_mask(batch_array, dense_mask)
    batch_label = tf.concat([batch_ind, label_ind], 0)
    indices = tf.transpose(tf.reshape(batch_label, [2, -1]))
    shape = [batch_size, tf.reduce_max(label_lengths)]
    vals_sparse = gather_nd(labels, indices, shape)

    return tf.SparseTensor(tf.to_int64(indices), vals_sparse, tf.to_int64(label_shape))



