import tensorflow as tf

def load_graph(model_file):
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

def print_operations(model_file):
    # Load the graph from the file
    graph = load_graph(model_file)
    
    # Print operations
    for op in graph.get_operations():
        print(op.name)

# Path to the classify_image_graph_def.pb file
model_path = 'classify_image_graph_def.pb'
print_operations(model_path)
