import sys
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants


K.set_learning_phase(0)
K.set_image_data_format('channels_last')

INPUT_MODEL = sys.argv[1]
NUMBER_OF_OUTPUTS = 1
OUTPUT_NODE_PREFIX = 'output_node'
OUTPUT_FOLDER= 'frozen'
OUTPUT_GRAPH = 'frozen_model.pb'
OUTPUT_SERVABLE_FOLDER = sys.argv[2]
INPUT_TENSOR = sys.argv[3]


try:
    model = load_model(INPUT_MODEL)
except ValueError as err:
    print('Please check the input saved model file')
    raise err

output = [None]*NUMBER_OF_OUTPUTS
output_node_names = [None]*NUMBER_OF_OUTPUTS
for i in range(NUMBER_OF_OUTPUTS):
    output_node_names[i] = OUTPUT_NODE_PREFIX+str(i)
    output[i] = tf.identity(model.outputs[i], name=output_node_names[i])
print('Output Tensor names: ', output_node_names)


sess = K.get_session()
try:
    frozen_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), output_node_names)    
    graph_io.write_graph(frozen_graph, OUTPUT_FOLDER, OUTPUT_GRAPH, as_text=False)
    print(f'Frozen graph ready for inference/serving at {OUTPUT_FOLDER}/{OUTPUT_GRAPH}')
except:
    print('Error Occured')



builder = tf.saved_model.builder.SavedModelBuilder(OUTPUT_SERVABLE_FOLDER)

with tf.gfile.GFile(f'{OUTPUT_FOLDER}/{OUTPUT_GRAPH}', "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

sigs = {}
OUTPUT_TENSOR = output_node_names
with tf.Session(graph=tf.Graph()) as sess:
    tf.import_graph_def(graph_def, name="")
    g = tf.get_default_graph()
    inp = g.get_tensor_by_name(INPUT_TENSOR)
    out = g.get_tensor_by_name(OUTPUT_TENSOR[0] + ':0')

    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def(
            {"input": inp}, {"outout": out})

    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map=sigs)
    try:
        builder.save()
        print(f'Model ready for deployment at {OUTPUT_SERVABLE_FOLDER}/saved_model.pb')
        print('Prediction signature : ')
        print(sigs['serving_default'])
    except:
        print('Error Occured, please checked frozen graph')

