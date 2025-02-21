import onnx
import sys

def remove_suffix_from_names(model_path, output_model_path, suffix=':0'):
    # Load the ONNX model
    onnx_model = onnx.load(model_path)

    # Get input and output names to remove the suffix from
    graph_input_names = [input.name for input in onnx_model.graph.input]
    graph_output_names = [output.name for output in onnx_model.graph.output]

    print('graph_input_names =', graph_input_names)
    print('graph_output_names =', graph_output_names)

    # Remove suffix from input names
    for input in onnx_model.graph.input:
        input.name = input.name.removesuffix(suffix)

    # Remove suffix from output names
    for output in onnx_model.graph.output:
        output.name = output.name.removesuffix(suffix)

    # Remove suffix from node input and output names
    for node in onnx_model.graph.node:
        for i in range(len(node.input)):
            if node.input[i] in graph_input_names:
                node.input[i] = node.input[i].removesuffix(suffix)

        for i in range(len(node.output)):
            if node.output[i] in graph_output_names:
                node.output[i] = node.output[i].removesuffix(suffix)

    # Save the modified ONNX model
    onnx.save(onnx_model, output_model_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 script.py <input_model.onnx> <output_model.onnx>")
        sys.exit(1)

    input_model_path = sys.argv[1]
    output_model_path = sys.argv[2]

    remove_suffix_from_names(input_model_path, output_model_path)